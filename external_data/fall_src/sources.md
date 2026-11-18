# 墜落防止補充資料來源

`external_data/fall_labels.tsv` 的圖片從下列公開文件抽出，依文件本身的圖說（錯誤例／改善例、缺失說明）標註。
圖檔與 PDF 不進 git，重建方式：下載 PDF 到 `fall_src/pdf/`，用 PyMuPDF 抽出 tsv 列出的頁面圖片（檔名 `<前綴>_p<頁>_<序>.jpg`）。

| 前綴 | 檔名 | 文件 | 網址 |
|---|---|---|---|
| tc | tc_audit.pdf | 臺中市政府工程施工查核常見缺失改善暨預防手冊（第一版） | https://www.rdec.taichung.gov.tw/media/352213/ |
| cp | coapre_manual.pdf | 財團法人職業災害預防及重建中心 營造業危害預防輔導手冊 | https://www.coapre.org.tw/files/202311151029056387.pdf |
| kr | kcg_roof.pdf | 高雄市 屋頂作業墜落預防安全指引 | https://orgws.kcg.gov.tw/001/KcgOrgUploadFiles/405/relfile/73873/205321/5d119a17-8786-4e96-abdb-a08937c9b8df.pdf |
| os | osha_spring.pdf | 職安署 107 年春安聯合稽查照片（臺南市美術館統包工程） | https://www.osha.gov.tw/media/xrohjupi/ |
| oc | osha_case112.pdf | 職安署 112 年營造業墜落重大職業災害案例彙整 | https://www.osha.gov.tw/media/25lawohq/ |
| of | osha_fall.pdf | 職安署 營造業墜落災害預防（摺頁） | https://taipei.ardswc.gov.tw/Content/Files/Article/6be803fb939145348c46dd0a6e244569.pdf |
| opl | osha_platform.pdf | 職安署 使用施工構台作業災害預防（摺頁） | https://taipei.ardswc.gov.tw/Content/Files/Article/e08cb3fa411a4fb1a6a630b77b098da3.pdf |
| ch | chc_audit2.pdf | 彰化縣政府 施工查核常見缺失之改善（預防）參考手冊（110 年修訂版） | https://ctrl.chcg.gov.tw/files/17_20211228092247798_003施工查核缺失改善(預防)參考手冊_2025.pdf |

刻意排除：外部測試集 ext113 的 4 份來源（新北市勞檢處 營造業安全衛生檢查重點宣導、高雄市勞檢處 營造作業危害辨識與災害預防、
管溝開挖擋土支撐作業安全指引、高雄市都發局 落實工地職業安全衛生管理作業手冊），避免訓練與測試重疊。
