param([string]$Docx, [string]$Pdf, [string]$Final)
# Открыть docx в Word, обновить оглавление и поля, сохранить итоговый docx и PDF, вывести число страниц.
$ErrorActionPreference = "Stop"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx, $false, $false, $false)
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Fields.Update() | Out-Null
    $pages = $doc.ComputeStatistics(2)
    if ($Final) { $doc.SaveAs2($Final, 16) }
    $doc.SaveAs2($Pdf, 17)
    $doc.Close($false)
    "pages=$pages"
} finally {
    $word.Quit()
}
