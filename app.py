"""Point d'entree Hugging Face Spaces (SDK Streamlit).

HF Spaces lance automatiquement `streamlit run app.py` a la racine ; on delegue au dashboard.
"""
import runpy

runpy.run_path("app/dashboard.py", run_name="__main__")
