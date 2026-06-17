### IMPORTANT
1. BEFORE IMPLEMENTING OR PLANNING ANYTHING, Make sure to ask user to do git pull from "dev" branch and look out for the already available implementation by others and check if it can be re-used. Must try to call the already available implementations and integrate them. Do not create duplications of the code.
2. BEFORE CHECKING (1), DO NOT START ANY IMPLEMENTATION OR PLANNING

CLAUDE CODE Best and Clean Coding Practices:

	1. In the python code, keep all the imports to the top (do not insert import statements in the middle of the code)
	2. While writing the python code, do not use too many variable names starting with _ or __ unless needed. 
	3. Always use descriptive variable names (avoid single letter variable name) even if it's lengthy - it's fine. 
	4. Writing multiple if-else ladders should be avoided. Create a config folder, design a json file
	and read the map from the json file - explain what functions to be called when a condition is satisfied.
	5. Avoid hardcoding variables to the best possible extent. Move them to the config.ini file and reference the variables from the config.ini file. E.g. any lists, dictionaries with hardcoded values  should be moved here to the config.ini file.
	6. Do not create too many config.ini files to implement (3) and  (4). Have one and use it globally.
	7. Do not hardcode any paths inside the code. Use a config.ini and put the paths there. For multiple scripts, create sections in the same config.ini. There should be only one config.ini per project. 
	8. Do not use try-except while importing modules in Python. Always make sure that the module is importable. If not , ask user for the installation
	9. Use classes and OOP even when the script is simple. And use class variables only to read common data among the methods.
	10. When you write docs (or .md) for scripts/sessions. Always create a folder called docs/ and put them there and add this docs/ folder to .gitignore except for README.md which should be git tracked.
	11. Do not use unicode characters in the code or print statements in the code. It should be ascii.
	12. In python, keep the import statements separate. In between the import statements, do not add conditional logic.
	13. Any intermediate python scripts/test scripts that you write, which is standalone from the codebase must be in claude_scripts folder. Create a folder called claude_scripts and put the scripts there. Add this folder to .gitignore
	14. Do not try to commit any of your changes unless explicitly asked.
	15. In scripts, always add verbose logging statements and give a switch -v to turn on for debugging. 
	16. Avoid variable names with single letters in codes. Need intuitive variable names
	17. Always do mkdir -p for any output dir that will be used by the code.
	18. Do not use arrows for prompts or print statements . Always use =>. Avoid robotic characters. 
	19. Do not keep writing documentation (.md) file of the fixes done and waste tokens, unless the user explicitly prompts it. Gemini CLI does not do this.
	20. Do not touch the code unnecessarily which is irrelevant to the logic that you are implementing
	21. Do add # comments to explain the critical code changes.
	22. For the python binary to be used, CREATE PYTHON="" in config.ini and use only that python for all your work. 
	23. Try hard to avoid an if-else condition in case of a fix, resort to if-else ladder only if you cannot fix it in a different way. Use hash maps if possible. 
	24. Do not generate any plots, txt files, md files in the current dir. Always organize them into plots, docs or claude_outputs folder
	25. When in plan mode, ALWAYS write the final plan into plans/<plan_name>_<timestamp>.md 
	26. SPAWN an agent always to run tests for the code and move on to the next tasks incase of multiple tasks.
	27. Avoid hardcoding defaults to CLI arguments unless the user explicitly prompted. Keep the default to None and error out if the arg is not passed. 
	28. Try to reuse code/function as much as possible. Before even starting to write the function, grep for the similar functions if exist and see if the function can be modified or updated by passing a switch.
	29. Must do typing (datatype) in python. Must properly type all the dummy arguments and the function output.
