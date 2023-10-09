@CALL "C:\Users\ken_nakatsukasa\Anaconda3\Library\bin\conda.bat" activate myWorkspace1
cd C:\Users\ken_nakatsukasa\Desktop\maimate_daily_report
python mm-data-extraction_15.py  -d . --market_dir=M:\csv1.5 --config_file=.\env.yml 0 > M:\\log.txt 2>&1
