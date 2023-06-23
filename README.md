# multimillionstartup
An AI Startup in 45 Minutes.

Demo! 


Wie funktionieren gute AI Startups? 
Im Beispiel: Dinge, bei denen viele Stunden teure Arbeit eingespart werden können. Wie zum Beispiel Rechtsanwälte.

Also machen wir ein Beispiel, was für die CloudLand passt, und machen teure
Routineaufgaben automatisch. 
Konkret Cloud Native Consulting. Cloud Native Consultants wie wir brauchen 
viel Zeit und sind teuer, nur weil sie ein paar Dinge wissen. 

Also automatisieren wir das. 

# Achtung! 
Der Bot ist eine in 60 Minuten entstandene Demo. Deshalb bekommt er Shell Access mit Zugriff auf kubectl, und damit Zugriff auf Kubernetes. 
Also bitte nur auf einen lokalen MiniKube oder so anwenden, und nicht gleich mit vollen Adminrechten gegen den Produktionscluster.

# Vorbereitung

https://platform.openai.com/account/api-keys

```shell
EXPORT OPENAI_API_KEY=sk-.....
```

https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html



```shell
conda create -n multimillionstartup python=3.10.9
conda activate multimillionstartup

pip install -r requirements.txt
```




s