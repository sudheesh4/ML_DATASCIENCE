import google.generativeai as genai
from selenium import webdriver
import time
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

API=""

genai.configure(api_key=API)




url = "https://www.themarginalian.org/2019/02/25/love-and-saint-augustine-hannah-arendt/"
def gettext(url):
    try:
        op = webdriver.ChromeOptions()
        op.add_argument('headless')
        testdriver = webdriver.Chrome(options=op)
        testdriver.get(url)
        time.sleep(5)
        
        body = testdriver.find_element(By.TAG_NAME,"Body")
        
        bodyc=BeautifulSoup(body.get_attribute("innerHTML"))
        #print(divbs)
        transcript=bodyc.get_text()
        #print(len(transcript))
        testdriver.quit()
        #print(">>>Summarising")
        #response=getsummary(transcript)
        response=transcript
    except:
        response="ERROR!"

    return response
    
def getsummary(data):
    window=4000
    i=0
    summary=''
    while i<len(data):
        text=data[i:i+window]
        temp=querytext(model,"Summarise the following : "+text)
        summary += temp
        i= i+window
    return summary
        

def getmodel():
    return genai.GenerativeModel("gemini-pro")

def querytext(mdl,prompt):
    #print(prompt)
    response=mdl.generate_content(prompt)

    try:
        res=response.text
    except:
        res=response.prompt_feedback
    return (res)  

def handleintent(model,userprompt,intent):
    tmanager=f"""Take a look at user's query with the intent of {res}, to assist user. Resolve the query if it is simple and clear. 
    If it is not clear or requires multiple steps and details, then ask relevant questions in order to respond to query appropriately.
    User: {userprompt}"""
    print(">>>>>"+intent)
    if intent.find("text")>=0:
        print("Texting!")
        tmanager=f"""Take a look at user's query and assist the user. Resolve the query if it is simple and clear. 
        If it is not clear or requires multiple steps and details, then ask relevant questions in order to respond to query appropriately.
        User: {userprompt}"""
        
        fin=querytext(model,tmanager)
    elif intent.find("book")>=0:
        print("Booking!")
        tmanager=f"""Take a look at user's query  with the intent of booking to assist the user. Resolve the query if it is simple and clear. 
        If it is not clear or requires multiple steps and details, then ask relevant questions in order to respond to query appropriately.
        User: {userprompt}"""
        
        fin=querytext(model,tmanager)
    elif intent.find("reason")>=0:
        print("Reasoning!")
        tmanager=f"""Take a look at user's query  with the intent of reasoning to assist the user. Resolve the query if it is simple and clear. 
        If it is not clear or requires multiple steps and details, then ask relevant questions in order to respond to query appropriately.
        User: {userprompt}"""
        
        fin=querytext(model,tmanager)

    elif intent.find("search")>=0:
        print("Searching!")
        tmanager=f"""Take a look at user's query  with the intent of carrying out search to assist the user. Resolve the query if it is simple and clear. 
        If it is not clear or requires multiple steps and details, then ask relevant questions in order to respond to query appropriately.
        User: {userprompt}"""
        
        fin=querytext(model,tmanager)

    elif (intent.find("analyse")>=0) or (intent.find("analysis")>=0):
        print("Analysing!")
        tmanager=f"""Take a look at user's query  with the intent of analysis, to assist the user. Resolve the query if it is simple and clear. 
        If it is not clear or requires multiple steps and details, then ask relevant questions in order to respond to query appropriately.
        User: {userprompt}"""
        
        fin=querytext(model,tmanager)
        
    elif (intent.find("technical")>=0):
        print("Technical!")
        tmanager=f"""Take a look at user's query, to assist the user with technical request. Resolve the query if it is simple and clear. 
        If it is not clear or requires multiple steps and details, then ask relevant questions in order to respond to query appropriately.
        User: {userprompt}"""
        
        fin=querytext(model,tmanager)

    elif (intent.find("multi-step")>=0):
        print("Multi-step!")
        #"  Return a list of individual actions to execute in a sequential manner to assist the user."
        tmanager=f"""Take a look at user's query, to assist the user with the multi-step action. 
        Resolve the query if it is simple and clear and all information is provided. 
        If it is not clear or requires and details, then ask relevant questions in order to respond to query appropriately.
        User: {userprompt}"""
        
        fin=querytext(model,tmanager)

    elif (intent.find("summary")>=0):
        print("Summary!")
        fin=getsummaryfromlink(model,userprompt)

    else:
        print("Unknown!")
        tmanager=f"""Take a look at user's query and assist the user as best as possible. Resolve the query if it is simple and clear. 
        If it is not clear or requires multiple steps and details, then ask relevant questions in order to respond to query appropriately.
        User: {userprompt}"""
        
        fin=querytext(model,tmanager)

    return fin
      

def review(model,userprompt,fin):
    reviewer=f"""Take a look at the following user request and response. If there are questions in it for clarification, return
     'yes' otherwise return 'no'. Only return 'no' if there are no questions being prompted to the user.
     User:{userprompt}
     Response:{fin}"""

    ask= querytext(model,reviewer)
    return ask
    

        
def genericquery(model,userprompt):
    manager=f"""You are a manager. Take a look at the user's query, and not execute it, but determine it's intent.
    The intent has to be among the following:
        text based / booking/ reasoning / search / summary / analysis / technical / multi-step actions/ UNKNOWN
    
    If intent is not clear return UNKNOWN.
        
    Some examples are as follows. 
    User: Tell me about France.
    Intent: text based query.
    
    User: Make a reservation for 5 people at an Indian restraunt."
    Intent: booking query.
    
    User: How do you cook this recipe?
    Intent: text based query.
    
    User: Book me a cab.
    Intent: booking query.
    
    User: Generate a summary of http:\\example.com\article.html?q=12n12j
    Intent: summary query.
    
    User: Make an itinery.
    Intent: text based query.
    
    User: Find me the cheapest flight.
    Intent: Internet search query.
    
    User: Book the cheapest train.
    Intent: booking query.
    
    User:Write a poem.
    Intent: text based query.
    
    User:{userprompt}
    Intent:"""
    
    res=querytext(model,manager)
    
    fin=handleintent(model,userprompt,res)
    
    ask=review(model,userprompt,fin)

    #prompt,res,fin,ask
    return {"prompt":userprompt,
            "intent":res,
            "response":fin,
            "asking-user":ask}


def getsummaryfromlink(model,userprompt):
    ppp=f"""You are a proof-reader. Look at the user query, you donot have to open the link or extract data.
    You only have to return the link user mentioned. If there is no link mentioned, return NO-LINK.
    User:{userprompt}"""
    link=querytext(model,ppp)
    if link.find('NO-LINK')>=0:
        #print('h')
        summ=getsummary(userprompt)
    else:
        #print('l')
        summ=getsummary(gettext(link))
    #print(summ)
    return summ


def assist(model,userprompt):
    print(">>>>"+userprompt)
    result=genericquery(model,userprompt)
    print("$$$$$"+result['response'])
    while result['asking-user'].find("no")==-1:
        temp=input('\n')
        result=genericquery(model,result['prompt']+' '+temp)
        print("%%%%"+result['response'])

model=getmodel()


prompt="write a haiku"
prompt="We are four people. We want to go to Times-square. Book the cheapest cab."
#prompt="Search for reviews of the movie 'inception.'"
#prompt="Plan a weekend getaway"
#prompt="What are the steps to install a software program?"
#prompt="Book tickets for the upcoming concert."
prompt="Describe the process of photosynthesis."
#prompt=" Find me the best deals on smartphones."
#prompt="Reserve a budget hotel for my trip to London."
#prompt=" Schedule a meeting for tomorrow at 2 PM."
#prompt="Search for the latest technology news."
#prompt="Summarise http:\\meta.com"
#prompt='Summarise elaborate point by point : "https://www.quantamagazine.org/maths-game-of-life-reveals-long-sought-repeating-patterns-20240118/" '
#prompt='Summarise elaborate point by point :  '+text
#prompt="What does 2+2 equal?"
#prompt="What is 47*34?"
#prompt="Following is a table of data. Tell me what is mean and variane."
#prompt="I wrote an essay about transmodernism. It was rejected. How can I improve? what steps I can do to exeute?"
#prompt="Find me the closest store that sells rabbits. Give me directions to reach there. And suggest average price of rabbits in the area."
#prompt="Decribe a python program to print fibonacci"
#prompt="What is the latest stockprice of APPLE?"
prompt="We want to go to a dinner place, make booking for 3."
#prompt="How to cook butter-chicken?"
#prompt="Which is the country famous for waffles? How can I go there from New York? Can you book me a flight?"
#prompt="Which is recent Olympics venue in 2024? what is the city famous for? How can I go there from New York?Book me tickets for the flight and event."


assist(model,prompt)