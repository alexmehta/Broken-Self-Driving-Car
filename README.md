# Broken-Self-Driving-Car
HOW TO RUN

1. Create Python Virtual Environment in Terminal (replace DIRECTORY with your specified directory):
   { python -m venv DIRECTORY }
   
2. Activate Python Virtual Environment:
     For Linux and I THINK osx:
       { source DIRECTORY/bin/activate }
   
     For Windows:
       CMD:
         { DIRECTORY/Scripts/activate.bat }
       POWERSHELL:
         { DIRECTORY/Scripts/Actvate.ps1 }

3. Install Pytorch:
   Go to the website ( https://pytorch.org/ ) and scroll down.
   Choose the settings that match with your system, ensure that pip, python, and stable release is selected.
   Enter the command generated into terminal and wait till installation is complete.

4.Run broken program:
  To run the program use command below while in directory with python file
  { python cartrainer.py }
