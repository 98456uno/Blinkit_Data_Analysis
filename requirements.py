# requirements.py
# This script generates a requirements.txt file for Blinkit Streamlit Dashboard and installs the packages

with open("requirements.txt", "w") as f:
    f.write("streamlit==1.35.0\n")
    f.write("pandas==2.2.2\n")
    f.write("matplotlib==3.8.4\n")
    f.write("seaborn==0.13.2\n")
    f.write("Pillow==10.3.0\n")

print("✅ requirements.txt created successfully!")

# Optional: Uncomment to install packages directly
# import os
# os.system("pip install -r requirements.txt")
