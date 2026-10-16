@echo off
echo Digital Signature Verification System
echo ====================================
echo.
echo Installing required packages...
pip install -r requirements.txt
echo.
echo Starting the application...
echo Open your web browser and go to the URL shown below
echo.
streamlit run app.py
pause
