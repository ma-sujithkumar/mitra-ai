## GOAL:

To form a cleaned up codebase and complete the integration (both backend and UI) end-to-end for the whole codebase and create an end-to-end ML training pipeline powered by Agentic AIs

## NOTES:
1. The codebase has commits from people who have worked in different epics - epic 1, epic 2, epic 3, epic 4
2. Each epic or each agent/tool in the epic was considered as a standalone project while development. So, duplication of the code is inevitable
3. The style of folders (directory structure) in each epic is different and so the scripting style - Need a common dir structure (explained below)
4. The auth page and auth db code will be pushed later. 

## CONSTRAINTS:
1. Only use google-adk for agent/llm calls.

## MORE CONTEXT (ON THE REQUIREMENTS)
2. Read /home/sujithma/mitra/meeting_transcripts
3. dataset2Vec is a new addition - should be used for warm start (or) initial selection in model_selection agent
4. I've tried my best to add a set of pending tasks in /home/sujithma/mitra/docs/tasks.md (this is for your warm-start and not limited to this)
5. Read /home/sujithma/mitra/DESIGN_PLAN.md
6. You can build context with these - but whatever I tell in the integration spec is final.

## TERMINOLOGY
1. Tool workspace dir - <path where the tool is invoked>/mitra/<user_id>/<session_id>/<all_session_related_files_and_outputs>

## APPROACH
1. Follow a phased approach - firstly, stitch and cleanup the backend flows/agents and finish smoke testing, then integrate with UI elements and enhance the UI
2. Split these into subtasks and spawn agents to complete them.

## REQUIREMENTS (IN RANDOM ORDER):
1. Any Google ADK LLM call which is happening in any of the agents of any epic should be in a common python script - llm/client.py in the tool dir
2. Create a binary/executable which I can invoke in any dir - It's a self hosted application and create a mitra dir there
3. Need a requirements.txt encompassing the requirements of the entire codebase. 
4. SPAWN an agent to explore about fitting google-adk native orchestrator to run the whole pipeline
5. Every agent should have metadata.csv and minidata.csv embedded in the prompt. The script which calls these agents should read from that dir.
6. need additional scripts (not part of the pipeline) in scripts/
7. Any additional results can be moved to docs/<module name>
8. With --cli mode, I should be able to run without invoking the UI
9. Epic 2,3,4 modules should dump the required visualizations to be picked up and shown in the UI - Must use the ones which are already dumped and enhance more. - But in the UI, do not show these plots by default. let the user click a button and the plots to be popped up.
10. Put UI interfacable classes in all the modules of all the epics.
11. dataset2Vec is a part of epic 3 -> the incoming dataset needs to be queried with the trained encoder, embedding similarity needs to be done existing 120 and leaderboards
12. Whichever flow does not have dependancy or a chain - those should run in parallel to avoid timing issues. E.g. SHAP and hyperparameter tuning can run in parallel.
13. Every agent/script need to be independently runnable and testable - the log should print the command to run
14. The UI should show the agents status and user should be able to resume from any agent (context management)
15. In EPIC 3, number of model to select first should be user-controllable (default to 10) and put the option in advanced_settings in the setup UI
16. The invocation of the application should also have a trainable encoder (already pre-trained for 120 datasets), the dataset2Vec encoder loaded in the background. 
17. infer (16) and get the dataset - leaderboard mapping. once mitra is done, add the newly acquired dataset embeddings and the final leaderboard into our DB back.
18. The judge agent in epic 4 should talk to epic 3 - to pick some other ML models and try if many of them are rejected (max turns: 3) - It should be a config in the advanced settings in page 1.
19. ML model selection -> trainer -> inference + SHAP + overfitting analyzer + Optuna (parallel) -> Judge agent -> selects models and again looks at MLKit and calls model selection again (max turns can be restricted)
20. Need the overall token count of each agent in tool workspace dir
21. need good logging which should explain the flow easily in tool workspace dir
22. 


## STRICT RULES:
1. No absolute paths are allowed anywhere in the codebase. Use only relative paths. ./ will refer to the run dir in which the app/bin is invoked.

## FIXES REQUIRED
1.   In the codebase, instead of generating or creating a single  client.py  script dynamically, the client connections for different epics are resolved and instantiated as follows:

  ### 1. In Epic 1 (Schema & Ingestion Backend)

  Instead of writing separate client files on disk as originally discussed in the design meeting ( openai_client.py ,  gemini_client.py ), the backend resolves the LLM settings
  dynamically on startup or on a per-run basis via  LlmSettingsResolver  and instantiates the client connector directly in memory:

  • It uses Google ADK's  LiteLlm  connector inside metadata_gen_agent.py to talk to OpenAI, Gemini, or Anthropic models using the parameters ( model ,  api_key ,  api_base ) passed at
runtime.

Actually - The epic-1 script should generate the llm/client.py based on the user feedback and smoke test it 
2. 

## OUTPUTS
1. bin/mitra - executable to invoke the app (node)
2. bin/setup.sh - setup all the requirements necessary to run
3. frontend/ - all the frontend code should be here
4. backend/ - all the backend code should be here
5. backend/agents/ - all the agents .py scripts to come here
6. backend/prompts - all the prompts should be present here. No prompt should be appearing in .py scripts/ 
7. docs/<module_name> => put the spec.md, chats, any benchmark results (rename the files accordingly)
8. DB/encoder.db, parquet files, optuna db, dataset to leaderboard mapping
9. config/config.yaml -> the configuration parameters with clear comments for all the modules.
10. Put every parameter in config.yaml under advanced_settings in the page 1 (setup) in the UI. when the app is invoked, copy this config.yaml there and use that only.
11. 

