import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import time


all_teams = []
##2023 Seasson 
html = requests.get("https://www.pro-football-reference.com/years/2023/index.htm").text
soup = BeautifulSoup(html, features="html.parser")
standings_table = soup.select('#AFC')[0]

links = standings_table.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if '/teams/' in l]
team_urls = [f"https://pro-football-reference.com{l}" for l in links]

standings_table = soup.select('#NFC')[0]
links = standings_table.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if '/teams/' in l]
team_urls2 = [f"https://pro-football-reference.com{l}" for l in links]

for i in team_urls2:
    team_urls.append(i)


for team_url in team_urls:
    team_name = team_url.split("/")[-2]
    data = requests.get(team_url).text
    soup = BeautifulSoup(data, features="html.parser")

    team_data = pd.read_html(StringIO(data), match="Schedule & Game Results")[0]
    team_data["Team"] = team_name
    all_teams.append(team_data)
    time.sleep(5)


##2022 Season
html = requests.get("https://www.pro-football-reference.com/years/2022/index.htm").text
soup = BeautifulSoup(html, features="html.parser")
standings_table = soup.select('#AFC')[0]

links = standings_table.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if '/teams/' in l]
team_urls = [f"https://pro-football-reference.com{l}" for l in links]

standings_table = soup.select('#NFC')[0]
links = standings_table.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if '/teams/' in l]
team_urls2 = [f"https://pro-football-reference.com{l}" for l in links]

for i in team_urls2:
    team_urls.append(i)


for team_url in team_urls:
    team_name = team_url.split("/")[-2]
    data = requests.get(team_url).text
    soup = BeautifulSoup(data, features="html.parser")

    team_data = pd.read_html(StringIO(data), match="Schedule & Game Results")[0]
    team_data["Team"] = team_name
    all_teams.append(team_data)
    time.sleep(5)




stat_df = pd.concat(all_teams)
stat_df.to_csv("games.csv")