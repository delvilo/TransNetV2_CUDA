# TorchCodec CUDA 解碼 vs. FFmpeg：架構、效能與安裝全面評比

## 一、 核心結論速覽

| 評比維度 | FFmpeg (CLI / `ffmpeg-python`) | TorchCodec (CUDA Device) | 結論與優劣 |
| :--- | :--- | :--- | :--- |
| **能否完全取代？** | 獨立工具，功能最全面 | **工作流程可取代，底層不可** | TorchCodec 底層本質上即是封裝 FFmpeg C 函式庫 |
| **端到端推論速度** | 較慢（受限於 PCIe 往返傳輸與 Pipe 開銷） | **極快（GPU 顯存直通 Zero-Copy）** | **TorchCodec 大勝**（可提升數倍整體吞吐量） |
| **安裝與維護難度** | **極低**（下載單一靜態二進位檔即可） | **較高**（需對齊 PyTorch、CUDA、FFmpeg 動態庫） | **FFmpeg 大勝**（環境相容性極佳） |
| **尺寸限制彈性** | 嚴格受限於 NV12 偶數規範（$48 \times 27$ 報錯） | **無限制**（GPU 解碼後由 PyTorch Tensor 直接插值） | **TorchCodec 大勝** |
| **最佳適用場景** | 跨平台部署、Windows、輕量腳本、TensorFlow/ONNX | Linux 伺服器、純 PyTorch 訓練與推論管線 | 視部署架構而定 |

---

## 二、 TorchCodec 的本質：它真的能取代 FFmpeg 嗎？

### 1. 概念釐清：TorchCodec 底層依然是 FFmpeg
TorchCodec（由 Meta / PyTorch 官方團隊開發）**並不是重新寫一套解碼器**，它的本質是：
* **底層依賴**：直接調用系統內建的 FFmpeg C 函式庫（`libavcodec`, `libavformat`, `libavutil` 等）以及 NVIDIA 的硬體解碼 SDK（NVDEC）。
* **封裝目標**：取代過去透過 Python `subprocess` / `pipe` 呼叫外部 FFmpeg CLI 的低效方式，將解碼能力直接嵌入 PyTorch 的 C++ 執行期核心。

> **結論**：
> 在 **Python 程式碼的呼叫介面** 上，TorchCodec 可以完全取代 `ffmpeg-python`；但在 **系統環境相依** 上，它並不能擺脫 FFmpeg，反而對 FFmpeg 的動態函式庫有更嚴格的依賴。

---

## 三、 速度與效能深度對比

對於 AI 影片模型（如 TransNetV2），效能瓶頸往往不是 GPU 運算，而是 **資料傳輸（PCIe 頻寬）** 與 **CPU-GPU 同步開銷**。

### 1. 資料傳輸路徑對比

```
【傳統 ffmpeg-python 流程】：
[影片檔案] 
   └──> NVDEC 硬體解碼 (GPU)
           └──> hwdownload (複製回 CPU 記憶體) ───[PCIe 頻寬消耗 #1]
                   └──> stdout 管道傳遞至 Python
                           └──> np.frombuffer (NumPy)
                                   └──> torch.from_numpy().cuda() ───[PCIe 頻寬消耗 #2]
                                           └──> TransNetV2 推論 (GPU)

【TorchCodec 流程】：
[影片檔案] 
   └──> NVDEC 硬體解碼 (GPU)
           └──> 原生封裝為 torch.Tensor (直接留在 GPU VRAM，零拷貝)
                   └──> torch.nn.functional.interpolate 縮放到 27x48 (GPU)
                           └──> TransNetV2 推論 (GPU)
```

### 2. 為什麼 TorchCodec 在速度上具有壓倒性優勢？
1. **完全零拷貝（Zero-Copy）**：影像在 GPU 解碼後，指標直接包裝為 `torch.Tensor`，資料全程留在顯存內，省去兩次昂貴的 PCIe 跨主機板傳輸。
2. **無跨程序 Pipe 開銷**：`ffmpeg-python` 依賴標準輸出管道傳輸原始位元組串流（Raw bytes），存在頻繁的程序間通訊（IPC）與記憶體配置開銷；TorchCodec 則是在同一記憶體空間內操作。
3. **完美解決 $48 \times 27$ 奇數解析度問題**：
   * 在 FFmpeg 端使用 `scale_cuda` 會因為 NV12 色度抽樣要求被迫為偶數而報錯。
   * 在 TorchCodec 中，解碼出原始畫質後，直接使用 PyTorch 的 `F.interpolate` 在 GPU 上將張量雙線性插值縮放到 $27 \times 48$，沒有任何尺寸限制，且速度遠快於 CPU 端的縮放。

---

## 四、 安裝難度與相容性對比

在安裝與環境建置層面，**TorchCodec 遠比 FFmpeg 繁瑣且容易踩坑**。

### 1. FFmpeg (`ffmpeg-python`) 的安裝難度：★☆☆☆☆
* **優點**：
  * 只需下載靜態編譯好的 `ffmpeg` 單一執行檔（例如 Gyan.dev 或 John Vansickle 提供的預編譯版本，內建 NVDEC/CUVID 支援），加到系統 `PATH` 即可。
  * Python 端只需執行純 Python 封裝套件：
    ```bash
    pip install ffmpeg-python
    ```
  * 完全不綁定 PyTorch 版本或 CUDA 驅動的 C++ 符號。

### 2. TorchCodec 的安裝難度：★★★★☆
* **痛點與限制**：
  1. **必須安裝 FFmpeg 動態連結共用函式庫（Shared Libraries）**：TorchCodec 無法直接讀取單一的 `ffmpeg.exe` 或靜態二進位檔，系統必須存在 `libavcodec.so`（Linux）或 `.dll`（Windows）。
  2. **CUDA / PyTorch 嚴格版號繫結**：TorchCodec 需對應特定的 PyTorch 與 CUDA Toolkit 版本（包含系統中的 `libnpp`, `libnvrtc`）。
  3. **平台支援度差異**：
     * **Linux**：官方 Wheels 支援最佳。
     * **Windows**：Windows 平台的 CUDA 硬體解碼支援限制較多，需手動處理動態連結庫路徑。

---

## 五、 Google Colab 環境實測：PyTorch / CUDA 13.0 一鍵安裝指南

在 **Google Colab（Linux Ubuntu 環境）** 下，TorchCodec **完全支援一鍵安裝**，而且比在 Windows 本機上順暢許多。

### 1. 為什麼 Colab 可以做到「一鍵安裝」？
* **系統共用庫已預載**：Colab 虛擬機已經預先安裝好 Ubuntu 官方的 FFmpeg 動態共用庫（包含 `libavcodec.so`, `libavformat.so` 等），免去了手動編譯 FFmpeg 的繁複手續。
* **Linux 原生 Wheel 支援**：PyTorch 官方對 Linux 平台提供了編譯好的 CUDA 預建 Wheel，`pip` 能自動抓取與 CUDA / PyTorch 版本匹配的二進位檔。

### 2. 一鍵安裝指令（針對 PyTorch 2.11 / CUDA 13.0）

在 Colab 筆記本的程式碼儲存格中執行以下指令即可：

```bash
# 一鍵安裝對齊 CUDA 13.0 的 TorchCodec
!pip install torchcodec --index-url=https://download.pytorch.org/whl/cu130
```

> **建議做法（確保 Torch 與 TorchCodec 版本完全同步）**：
> 若環境中的 PyTorch 尚未完全對齊 CUDA 13.0，建議使用聯合更新指令，避免 C++ ABI 不相容：
> ```bash
> !pip install --upgrade torch torchcodec --index-url=https://download.pytorch.org/whl/cu130
> ```

---

### 3. Colab 實測避坑清單（Checklist）

1. **必須開啟 GPU 執行階段**：
   * 進入選單：`執行階段 (Runtime)` $\rightarrow$ `變更執行階段類型 (Change runtime type)` $\rightarrow$ 硬體加速器選擇 **T4 GPU** 或 **L4 / A100 GPU**。
   * 若處於純 CPU 執行階段，初始化 `device="cuda"` 會拋出 `RuntimeError: No CUDA GPUs are available`。

2. **驗證 GPU 硬體解碼器（NVDEC）是否存在**：
   Colab 提供的 T4、L4、A100 等企業級顯卡均具備標準 NVDEC 晶片，可直接支援 H.264、HEVC 等主流格式的硬體解碼。

3. **版本衝突報錯排除**：
   若安裝後出現 `RuntimeError: Could not load libtorchcodec`，通常是因為 Colab 原先預載的舊版 `torch` 與新安裝的 `torchcodec` 發生版本衝突。此時只需：
   * 點選 Colab 選單：`執行階段 (Runtime)` $\rightarrow$ `重新啟動工作階段 (Restart session)`。
   * 重新執行 `import torchcodec` 即可正常載入。

---

### 4. 30 秒快速驗證腳本

安裝完成後，可以在 Colab 執行以下代碼測試 CUDA 硬體解碼是否就緒：

```python
import torch
from torchcodec.decoders import VideoDecoder

# 1. 檢查 PyTorch 與 CUDA 環境
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"當前 GPU: {torch.cuda.get_device_name(0)}")

# 2. 自動生成 1 秒測試影片
!ffmpeg -f lavfi -i testsrc=size=1920x1080:rate=30 -t 1 -pix_fmt yuv420p /tmp/test_video.mp4 -y -loglevel quiet

# 3. 測試 TorchCodec CUDA 解碼
decoder = VideoDecoder("/tmp/test_video.mp4", device="cuda")
frames = decoder.get_frames_at(indices=[0, 1, 2]).data

print(f"成功使用 CUDA 解碼！輸出張量形狀: {frames.shape}, 裝置: {frames.device}")
# 預期輸出: 成功使用 CUDA 解碼！輸出張量形狀: torch.Size([3, 3, 1080, 1920]), 裝置: cuda:0
```

---

## 六、 TransNetV2 實作對比示範

### 方案 A：使用 TorchCodec（純 GPU 管線，推薦 PyTorch 專案採用）

```python
import torch
import torch.nn.functional as F
from torchcodec.decoders import VideoDecoder

video_path = "input.mp4"

# 1. 初始化 GPU 解碼器
# seek_mode="approximate" 可大幅加速長影片的載入速度
decoder = VideoDecoder(video_path, device="cuda", seek_mode="approximate")
num_frames = decoder.metadata.num_frames

print(f"影片總幀數: {num_frames}, FPS: {decoder.metadata.average_fps}")

# 2. 批次解碼並直接在 GPU 上縮放為 TransNetV2 規格 (27, 48)
batch_size = 100
for i in range(0, num_frames, batch_size):
    end_idx = min(i + batch_size, num_frames)
    indices = list(range(i, end_idx))
    
    # 取得張量，形狀為 (Batch, Channels, Height, Width)，位於 cuda:0
    frames_tensor = decoder.get_frames_at(indices=indices).data
    
    # 直接在 GPU 進行雙線性插值縮放到 27x48（高度 27，寬度 48）
    # 無需擔心 NV12 偶數問題
    frames_resized = F.interpolate(
        frames_tensor.float(), 
        size=(27, 48), 
        mode="bilinear", 
        align_corners=False
    ).byte()
    
    # 轉為 TransNetV2 所需的 (Batch, 27, 48, 3)
    input_frames = frames_resized.permute(0, 2, 3, 1)
    
    # 直接傳入 PyTorch 版本的 TransNetV2 進行推論
    # with torch.no_grad():
    #     logits = model(input_frames)
```

---

## 七、 決策建議：您應該更換為 TorchCodec 嗎？

```
您是否應該改用 TorchCodec？
 │
 ├── 您的模型是否以 PyTorch 運行？
 │    ├── 否 (使用 TensorFlow / ONNX Runtime) ─────> 【維持 FFmpeg】(TorchCodec 綁定 PyTorch)
 │    └── 是
 │         │
 │         ├── 您的作業系統為何？
 │         │    ├── Windows ──────────────────────> 【維持 FFmpeg】(Windows CUDA 支援仍不穩定)
 │         │    └── Linux / Google Colab
 │         │         │
 │         │         └── 是否追求極致批次推論吞吐量 (High Throughput)？
 │         │              ├── 是 ─────────────────> 【強烈推薦更換至 TorchCodec】
 │         │              └── 否 (僅需偶爾處理短片) ─> 【維持現有 FFmpeg 偶數裁切方案即可】
```