# requirements.py
with open("requirements.txt", "w") as f:
    f.write("streamlit==1.46.1\n")
    f.write("pandas==2.2.2\n")
    f.write("matplotlib==3.10.3\n")
    f.write("seaborn==0.13.2\n")
    f.write("pillow==10.2.0\n")  # downgraded from 10.3.0 to avoid build error
print("✅ requirements.txt created successfully!")

# Optional install line
# import os; os.system("pip install -r requirements.txt")
