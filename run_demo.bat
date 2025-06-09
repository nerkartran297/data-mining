@echo off
echo =================================
echo    DATA MINING PROJECT DEMO
echo =================================
echo Starting Streamlit application...
echo Open your browser at: http://localhost:8501
echo.
@REM pip install streamlit pandas numpy matplotlib seaborn scikit-learn mlxtend minisom graphviz
streamlit run app.py
pause 