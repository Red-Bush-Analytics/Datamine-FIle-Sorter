This Repository contains scripts to classify and sort Datamine project files.<br>
<br>
The repository contains three Python scripts,  a Word document with build history and instructions for using the scripts,  and an example of how to set up and run the summary-classifier-sorter in Jupyter Notebook.<br>
<br>
A Datamine license is not necessary to run these scripts and access the Files.<br>
<br>
How it works: <br>
- The file Datamine_DM_File_Workflow_Documentation_jupyter_organiser_final.docx contains information about the build and instructions on how to use the scripts in the command line and Jupyter Notebook. Use an IDE for debugging and editing.
- The first script (datamine_dm_legacy_reader_working.py) reads the .dm binary files and creates a summary with the header information. An option to export to CSV/xlxs is included.
- Using the header information, the second script (datamine_file_classifier_working.py)  classifies the files into 36 file types, with an unknown category for non-standard file types.
- The third script (datamine_file_organiser_working.py)  sorts the files into folders.
- Files in the Unknown folder will need to be sorted manually

Why three scripts? 
1. It is necessary to verify that the output is correct between the steps.
2. There are many combinations possible, so modular works well.
3. That is as far as I have gotten at this point! 

These scripts are specifically set up for Datamine; however, they should be adaptable to any other software that stores its files as binary or text.


The reader was trained on data files produced over the last 20 years for Datamine Studio, Studio2, Studio, EM, RM, RM2, and RM2+. It should read most legacy .dm files.  <br>
I have not formatted all file types; I have only formatted the ones I use regularly.  If you have additional inputs/outputs that are not in the classifier suite, you will need to modify the code.<br>
<br>
Known Issues:<br>
- Sometimes there are extra lines at the end of the csv/xlsx.  It seems to happen with "Compact" file types. Mostly, these lines are garbage and can be deleted, but check.  If it happens regularly, you have a file structure that the code does not recognise, and you will need to modify the code.  I recommend using ChatGPT for this, as decoding binary files is a pain, and the AI does it faster than you can.  You will need to modify the code to read the new file type.
- If your .dm files contain non-standard characters, the parser may not correctly decode spacing/columns. You need to correct your data or edit the reader to read these characters.
- There is still an issue reading some CSV files from TONGRAD.  It does not always read TONGRAD/TIV files that have been modified after output. It does not like some TIV files from TONGRAD.
