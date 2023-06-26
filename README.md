# Cloud Consultant AI

At [Cloudland 2023](https://www.cloudland.org/en/home/) we encountered a lot of interest in AI topics as well. 
To demonstrate how easy it is to write simple AI bots using langchain we did a fast evening session to provide a bot that helps with cloud issues. 

The two major features of the bot are:
* knowledgebase for cloud native and kubernetes topics - just ask him anything about it. 
* ability to directly speak to kubernetes and / or aws using kubectl / aws cli. 


![Demo Screengrab](demo.gif "Demo")


## Important

This bot was developed within 60 minutes, so don' expect to much. 
Obviously there is some danger hidden in the fact that it directly uses your shell to call kubectl, aws and helm cli. 
However: before any command is executed you need to confirm the command. 

This is alpha quality democode. Don't expect it to actually replace your cloud consultant.

# Preparation

Please get yourself an OpenAi API key [here](https://platform.openai.com/account/api-keys).
It uses GPT-3.5-turbo.

Make the API key available in your environment:

```shell
EXPORT OPENAI_API_KEY=sk-.....
```

The bot is based on python and tested with python 3.10. 
I recommend to use conda or virtualenv to properly separate python environments.
You'll find the information needed to install conda 
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)



```shell
conda create -n cloudbot python=3.10.9
conda activate cloudbot
pip install -r requirements.txt
```

Start the bot on the command line:
```shell
python bot.py
```




s