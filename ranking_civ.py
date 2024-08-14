from math import sqrt, exp
import numpy as np
import matplotlib.pyplot as plt
import random as rd
GROUP = [["Rémy",4,3], ["Antoine",4,4], ["Matthieu",3,3], ["Yanis",3,4], ["Amaury",2,3], ["Joseph",4,4], ["Dorian",1,2], ["Thomas",2,1]]
VICTORIES = ["Religious", "Military", "Territorial", "Scientific", "Cultural", "Diplomatic", "Score"]
total_group_size = 8

def putting_logistic_function(average_elo,number,logistic_parameters,elo_difference):
    K = logistic_parameters[0]
    a = logistic_parameters[1]
    r = logistic_parameters[2]
    return (average_elo/number)/(1+a*exp(-r*elo_difference))

def redistribution_ratios(n):
    ratios = np.zeros(n)
    unit_factor = 0
    for k in range(2,n+2):
        unit_factor += (1/k)**2
        ratios[k-2] = (1/k)**2  
    return ratios/unit_factor 

def data_low_pass(x,y,window):
    n = len(x)
    avg_y = []
    for i in range(n):
        avg = 0
        if i < window:
            for j in range(i+1):
                avg += y[j]
            avg = avg/(i+1)
        elif n-i < window:
            for j in range(n-i+1):
                avg += y[n-j-1]
            avg = avg/(n-i+1)
        else:
            for j in range(i,i+window):
                avg += y[j]
            avg = avg/window
        avg_y.append(avg)
    return avg_y
                
class Player:
    
    def __init__(self, name, elo, strength, presence):
        self.name = name
        self.elo = elo
        self.strength = strength
        self.presence = presence
        self.results = []
    
    def add_elo(self, addition):
        self.elo += addition
    
    def display(self):
        print(f"Je suis {self.name}.")
        print(f"Mon élo est de {self.elo}.")
        
    def generate_score(self):
        score = rd.gauss(200*self.strength,100)
        if score < 0:
            return 0
        else:
            return score
        
class Result:
    
    def __init__(self, player, score, victory):
        self.player = player
        self.score = score
        self.victory = victory
    
    def display(self):
        if self.victory == "Lose":
            print(f"{self.player.name} a perdu avec un score de {self.score}.")
        else:
            print(f"{self.player.name} a gagné avec un score de {self.score}.")
    
class Game:
    
    def __init__(self, players):
        self.players = players
        size = 0
        average_elo = 0
        for player in self.players:
            average_elo += player.elo
            size += 1
        self.average_elo = average_elo/size
        self.size = size
        self.results = []
        
    def add_result(self, result):
        self.results.append(result)
    
    def display_results(self):
        print("----- Voici les résultats de la partie -----\n")
        for result in self.results:
            result.display()
        print("\n")
    
    def get_leaderboard(self):
        self.display_results()
        for result in self.results:
            if result.victory != "Lose":
                winner = result.player
                self.results.remove(result)
        leaderboard = [winner]
        self.results = sorted(self.results, key=lambda result:result.score, reverse = True)
        for result in self.results:
            leaderboard.append(result.player)
        return leaderboard
    
    def display_leaderboard(self, leaderboard):
        i = 1
        for p in leaderboard:
            print(f"{i}. {leaderboard[i-1].name}")
            i+=1

    def update_elo(self,logistic_parameters):
        if self.size<(int(total_group_size/2)+1):
            return
        else:
            sample_factor = sqrt(self.size/total_group_size)
            leaderboard = self.get_leaderboard()
            rank = 1
            self.display_leaderboard(leaderboard)
            total_putting = 0
            for player in leaderboard:
                elo_difference = player.elo - self.average_elo
                putting = sample_factor*putting_logistic_function(self.average_elo,self.size,logistic_parameters, elo_difference)
                if putting < player.elo:
                    player.add_elo(-putting)
                    total_putting += putting
                else:
                    total_putting += player.elo
                    player.add_elo(-player.elo)
                    
            unit_factor = 0
            for k in range(2,self.size+2):
                unit_factor+=1/k
            print(unit_factor)
            for player in leaderboard:
                player.add_elo((total_putting/(rank+1))/unit_factor)
                rank+=1
     
class Simulation:
    
    def __init__(self, name, group, nb_games, logistic_parameters):
        self.name = name
        self.players_pool = []
        for person in group:
            p = Player(person[0], 500, person[1], person[2])
            self.players_pool.append(p)
        self.nb_players = len(self.players_pool)
        self.nb_games = nb_games
        self.played = 0
        self.data = np.zeros((self.nb_players,self.nb_games))
        self.logistic_parameters = logistic_parameters
    
    def play_game(self):
        players = []
        for p in self.players_pool:
            x = rd.random()
            if x>(1/(p.presence+1)):
                players.append(p)
        game = Game(players)
        chances = []
        for p in players:
            chances.append(p.strength**3)
        winner = rd.choices(players, weights=chances, k = 1)[0]
        result = Result(winner, 2000*rd.random(), rd.choice(VICTORIES)[0])
        game.add_result(result)
        
        for player in players:
            if player != winner:
                result = Result(player, player.generate_score(), "Lose")
                game.add_result(result)
        print("La partie a été jouée.")
        game.update_elo(self.logistic_parameters)
        print("Les scores élos des joueurs concernés ont été mis à jour.")
        self.played+=1
    
    def update_data(self,k):
        i = 0
        for p in self.players_pool:
            self.data[i][k] = p.elo
            i+=1
        
    def play_simulation(self):
        for i in range(self.nb_players):
            self.data[i][0] = 500
        for k in range(1,self.nb_games):
            self.play_game()
            self.update_data(k)
    
    def display_simulation_data(self):
        games = np.linspace(0,self.nb_games, self.nb_games+1)
        for k in range(self.nb_players):
            plt.plot(games[:-1], self.data[k], label = GROUP[k][0])
        plt.legend()
        plt.show()
        
    def display_smoothed_data(self,window):
        games = np.linspace(0,self.nb_games, self.nb_games+1)
        for k in range(self.nb_players):
            plt.plot(games[:-1], data_low_pass(games[:-1], self.data[k], window), label = GROUP[k][0])
        plt.legend()
        plt.show()
        
    def display_putting_function(self):
        x = np.linspace(-500,500,5000)
        y = np.zeros(5000)
        for k in range(5000):
            y[k] = putting_logistic_function(self.logistic_parameters,x[k]) 
        plt.plot(x,y)
        plt.show()
         
NUMBER_OF_GAMES = 10000
sim = Simulation("premier_test",GROUP,NUMBER_OF_GAMES,[10,1,5*10e-4]) 
print(sim.players_pool[0].name)   
sim.play_simulation()
sim.display_simulation_data()
sim.display_smoothed_data(NUMBER_OF_GAMES//100)