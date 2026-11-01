# PDF 翻譯器

只翻譯 PDF 的文字並維持原始版面：譯文寫回原文字的位置，基線、字級、顏色、對齊方式都沿用原文，所有譯文統一使用同一套字型（字型正規化）。圖片、向量圖形、表格框線、連結都不會動到。

## 功能

- **版面保留**：抽出段落後移除原文字形（不加白底），再把譯文排進同一個框。放不下時先等比縮小字級（下限可設定），最後才水平壓縮。
- **動態學習版面規則**：翻譯過程中從文件累積兩條規則，輸出時套用到全文件。
  - 樣式縮放：同一樣式（原字級＋粗細）的文字使用同一縮放比例，取該樣式所有段落中能放進原框的比例的第 20 百分位，因此各頁同類文字大小一致。只有比例更低的離群段落會再個別縮小。
  - 長度比：統計譯文與原文的長度比。預估會超出原框的多行段落，會在翻譯時附上「版面容量（字數上限）」。
  - 精簡：翻完後仍放不下的多行段落，會先請翻譯模型精簡一次，然後才進入檢查。
- **只重新排版，不重新翻譯**：譯文都存在進度檔裡。翻完後如果發現字型、縮放或字距不理想，在「排版 / 字型」分頁改設定，再按「重新排版輸出（不翻譯）」。程式會用新設定重新學習樣式縮放規則並輸出，不呼叫 API。log 會列出有幾段仍需額外縮小或水平壓縮，方便判斷要再怎麼調。暫停中也可以先按這個按鈕預覽新設定，按「繼續」後才會套用到後續翻譯。
- **字型正規化**：預設 Noto Sans CJK TC 常規和粗體，也可以選 Noto Serif CJK TC、微軟正黑體、PyMuPDF 內建字型或自訂字型檔。輸出時會自動做字型子集化，檔案不會變大。
- **UI**（tkinter）：可以暫停，暫停後修改模型、思考程度、併發數、批次大小等設定，按「繼續」立即套用。
- **併發翻譯**：可分別設定「同時處理頁數」和「最大並發請求數」，執行中調整也會生效。
- **中斷恢復**：每段翻譯都即時存進 `work/<檔名>_<hash>_<語言>/state.json`。停止、當機或關閉程式後，再按「開始」會從中斷處繼續。
- **多步驟翻譯檢查**：一頁翻完後，由獨立的檢查模型模糊比對全頁原文和全頁譯文，只檢查有沒有跑題、錯譯、漏譯。檢查模型只回答通過與否，加上被否決的段落區間（標註所屬批次）。被否決的區間會連同全頁原文退回翻譯模型，要求只修復該段，修完再重新檢查，最多重複「最多修復輪數」次。
- **模型分開設定**：翻譯和檢查可以使用不同供應商、模型與思考努力程度。
  - `anthropic`：Claude（預設翻譯 `claude-opus-5-5`、檢查 `claude-sonnet-5-5`），思考程度 `low`～`max`。
  - `openai`：任何相容 OpenAI Chat Completions 的服務，填 Base URL 即可（OpenAI、DeepSeek、OpenRouter、本地 vLLM/Ollama 等），思考程度會以 `reasoning_effort` 傳送。

## 安裝

需要 [uv](https://docs.astral.sh/uv/)：

```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```sh
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

在專案根目錄建立 `.env`：

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...        # 可選
DEEPSEEK_API_KEY=...         # 可選；在 UI 的「API Key 環境變數」欄位填 DEEPSEEK_API_KEY
```

## 使用

- Windows：雙擊 `start.bat`
- 其他系統：`./start.sh`
- 或直接執行：`uv run pdf-translator`

第一次執行時 uv 會自動建立虛擬環境並安裝相依套件。

### 命令列模式（不開 UI）

```
uv run python -m pdf_translator.cli 輸入.pdf -o 輸出.pdf --pages 1-5 [--no-check] [--target English]
```

其餘設定會讀取 UI 存下的 `config.json`。

只重新排版已翻好的檔案（不呼叫 API）：

```
uv run python -m pdf_translator.cli 輸入.pdf -o 輸出.pdf --relayout [--font "微軟正黑體"] [--spacing 0.03] [--min-scale 0.6] [--max-scale 1.1]
```

### OpenAI 相容服務設定範例

| 服務 | Base URL | 模型範例 |
|---|---|---|
| OpenAI | （留空） | 依帳號可用模型 |
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat` |
| OpenRouter | `https://openrouter.ai/api/v1` | `anthropic/claude-sonnet-5-5` |
| Ollama | `http://localhost:11434/v1` | `qwen3` |

如果供應商不支援 `response_format`，請取消勾選「JSON mode」；不支援 `reasoning_effort` 時，思考程度選 `default`。

## 設定重點

| 設定 | 說明 |
|---|---|
| 每批最大字元數 / 段落數 | 一頁內的段落會切成多批送出，每批都附上全頁原文作為上下文（Claude 會快取這部分）。 |
| 同時處理頁數 / 最大並發請求數 | 遇到速率限制（429）時調低。程式會自動退避重試。 |
| 最多修復輪數 | 檢查否決後最多修復幾輪，達上限仍被否決的頁面會記錄在 log。 |
| 最小字級縮放 | 譯文放不下時，字級最多縮到原字級的這個比例。 |
| 最大字級縮放 | 預設 1.0（不放大）。設大於 1 時，較短的譯文可以放大填滿原框，同一樣式仍保持一致。 |
| 字距 | 每個字之間額外的間距，單位 em（字級的倍數），預設 0。負值為緊縮。不為 0 時會逐字繪製，輸出檔約大 2 倍。 |
| 動態學習版面規則 / 精簡過長譯文 | 對應上面兩個機制，可分別關閉。學到的規則會即時顯示在 UI 狀態列。 |
| 額外指示 / 詞彙表 | 會附加在翻譯模型的指示後面，例如專有名詞對照表。 |

遇到 API Key 錯誤、模型名稱錯誤或參數不支援時，工作會**自動暫停**。修正設定後按「繼續」即可。

## 目錄

```
src/pdf_translator/
  app.py         UI
  pipeline.py    工作流程：併發、暫停、中斷恢復、檢查與修復
  llm.py         Anthropic / OpenAI 相容 API 呼叫
  pdf_layout.py  段落抽取與譯文排版
  config.py      設定與字型
  cli.py         命令列模式
work/            翻譯進度（可刪除）
config.json      UI 設定（自動產生）
```

## 限制

- 只處理有文字層的 PDF。掃描檔需要先做 OCR。
- 直排文字或旋轉的文字保留原文不翻譯。整頁旋轉（/Rotate）的頁面未特別處理。
- 上標的註腳標號（例如 `1)`）會併入段落，以一般字級排出。
- 段落切分是依版面推測。若某頁切分不理想，可以在 `state.json` 檢查該頁的 `units`。
