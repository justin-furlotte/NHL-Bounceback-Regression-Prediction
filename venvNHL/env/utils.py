from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Plotting The Results

# Finds the top n overperformers and underperformers who played at least min_games games, 
# and then creates a bar plot showing 
# 
# 1) The model's predicted number of goals scored that season (in green).
# 
# 2) The actual number of goals scored that season (in gold).
# 
# 3) The actual number of goals scored the following season (in light blue).
# 
# 
# 
# - All goals are the player's *pace over 82 games*. 
# 
# - On the ***overperformer*** chart, if the blue bar (goals next season) is less than the 
# gold bar (goals last season), then the player's name is written in green, because the 
# model correctly predicted that the player would regress the following season. Otherwise, 
# the player's name is written in red, because the model made a mistake.
# 
# - Conversely, on the ***underperformer*** chart, if the blue bar is higher than the gold bar, 
# then the player's name is in green because the model successfully predicted that the player 
# would bounce back the following season.  Otherwise, the player's name will be in red.
# 
# By default, n=10, i.e. showing the top ten over/underperformers, 
# and min_games=40, i.e. the player must have played at least 
# 40 games to be considered for the chart. 

class Chart:

    def __init__(self):
        self.scatter_df = None

    def CreateScatterDF(self, model, dfs, min_games=40):

        scatter_df = []

        for i in np.arange(10,22):
            
            X = dfs[str(i)+"_"+str(i+1)].drop(columns="I_F_goals")
            y = dfs[str(i)+"_"+str(i+1)]["I_F_goals"]

            # Only consider the players who played min_games in both seasons
            y = y.loc[X['games_played'] >= min_games]
            X = X.loc[X['games_played'] >= min_games]

            y_hat = model.predict(X)

            y_this_season = np.array(y)
            games_played = np.array(X["games_played"])
            goal_pace = y_this_season/games_played*82

            pred_goal_pace = y_hat/games_played*82

            scatter_dictX = {
                "Season": [2000+i for j in range(len(X["team"]))],
                "Team": list(X["team"]),
                "Player": list(X["name"]),
                "Goal pace": list(goal_pace),
                "Predicted goal pace": list(pred_goal_pace)
                }

            scatter_dfX = pd.DataFrame(scatter_dictX)
            scatter_dfX.round(decimals=1)

            scatter_df.append(scatter_dfX)

        return pd.concat(scatter_df)


    # Comparing Predicted Goals and Actual Goals
    def UnderPerformer(self, differences_last_year): 
        
        underperformer_locs = np.argsort(differences_last_year)
        
        return underperformer_locs

    def OverPerformer(self, differences_last_year): 
        
        overperformer_locs = np.argsort(differences_last_year)[::-1] # Reversed order
        
        return overperformer_locs

    # Requires a fitted model.
    # Also, be careful that X_last_year is a subset of X_this_year,
    # For example use X_last_year test and X_this_year entire dataset.
    def Find(self,
            PerformerType, 
            X_last_year, 
            X_this_year, 
            y_last_year, 
            y_this_year, 
            model, 
            top_n = 10,
            min_games = 40,
            print_details=True,
            produce_plot=True): # Decorator function

        perf_type = None
        season = str(X_last_year["season"].iloc[0])+"-"+str(X_last_year["season"].iloc[0]+1)
        future_season = str(X_last_year["season"].iloc[0]+1)+"-"+str(X_last_year["season"].iloc[0]+2)

        print("\n")
        if PerformerType == self.UnderPerformer:
            perf_type = "Underperformers"
        elif PerformerType == self.OverPerformer:
            perf_type = "Overperformers"
        else:
            print("Incorrect PerformerType.")

        if print_details:
            print("-----------------------------------")
            print("Top", top_n, perf_type+" in "+season)

        # Only consider the players who played min_games in both seasons
        y_last_year = y_last_year.loc[X_last_year['games_played'] >= min_games]
        X_last_year = X_last_year.loc[X_last_year['games_played'] >= min_games]
        y_this_year = y_this_year.loc[X_this_year['games_played'] >= min_games]
        X_this_year = X_this_year.loc[X_this_year['games_played'] >= min_games]
        

        
        # Differences in actual goals minus predicted goals for last year
        y_pred_last_year = model.predict(X_last_year)

        games_played_last_year_all = np.ravel(X_last_year["games_played"])

        differences_last_year = np.ravel(y_last_year)/games_played_last_year_all*82 - np.ravel(y_pred_last_year)/games_played_last_year_all*82  
        
        # Sort the differences to identify under/overperformers. 
        # Earlier in the list means more under/overperformance last season.
        performer_locs = PerformerType(differences_last_year)

        ly = []
        lyp = []
        ty = []
        typ = []
        names = []
        n = 0
        i = 0

        while n < top_n:

            # Get some info about the under/overperformer
            j = performer_locs[i]
            player_id_last_year = pd.DataFrame(y_last_year.iloc[j]).T.index[0]
            this_year = y_this_year.index[0][8:]
            player_id_this_year = player_id_last_year[:7]+"_"+this_year

            name = pd.DataFrame(X_last_year.loc[player_id_last_year]).T["name"].iloc[0]
            
            # Find the games played, actual goals, and predicted goals from last year
            games_played_last_year = X_last_year["games_played"].loc[player_id_last_year]
            goals_last_year = y_last_year.loc[player_id_last_year][0]
            pred_goals_last_year = float(model.predict(pd.DataFrame(X_last_year.loc[player_id_last_year]).T)[0])

            # Make sure that player actually played next year
            if player_id_this_year in X_this_year.index.tolist():

                n += 1
                i += 1

                # Find the games played, actual goals, and predicted goals from this year
                games_played_this_year = X_this_year["games_played"].loc[player_id_this_year]
                goals_this_year = y_this_year.loc[player_id_this_year][0]
                pred_goals_this_year = float(model.predict(pd.DataFrame(X_this_year.loc[player_id_this_year]).T)[0])

                goal_pace_last_year = goals_last_year / games_played_last_year * 82
                pred_goals_last_year = pred_goals_last_year / games_played_last_year * 82
                goal_pace_this_year = goals_this_year / games_played_this_year * 82
                pred_goals_this_year = pred_goals_this_year / games_played_this_year * 82

                ly.append(goal_pace_last_year)
                lyp.append(pred_goals_last_year)
                ty.append(goal_pace_this_year)
                typ.append(pred_goals_this_year)
                names.append(name)

                if print_details:

                    # Let's see if they actually bounced back next season!
                    print("-----------------------------------")
                    print("\n")

                    print(i+1)
                    print("Player:", name)
                    print("Player ID:", player_id_this_year)
                    print("\n")

                    print(season+" Season")
                    print("----------------------")
                    print("Goal pace over 82 games: {:.2f}".format(goal_pace_last_year))
                    print("Predicted goal pace over 82 games: {:.2f}".format(pred_goals_last_year))
                    print("Games played:", games_played_last_year)
                    print("\n")

                    print(future_season+" Season")
                    print("----------------------")
                    print("Goal pace over 82 games: {:.2f}".format(goal_pace_this_year))
                    print("Predicted goal pace over 82 games: {:.2f}".format(pred_goals_this_year))
                    print("Games played:", games_played_this_year)
                    print("\n")
                
            else:
                i += 1
                print("Player", name, "(player ID", player_id_this_year, ") did not play in the "+future_season+" season.\n\n")
                print("Will skip ahead to next player.\n")
        # Produce plot if desired
        if produce_plot:

            x = np.arange(np.max([10,top_n]))
            width = 0.4

            def CorrectColor(t):
                if t == 1:
                    return "g"
                else:
                    return "r"

            fig, ax = plt.subplots()
            if perf_type=="Overperformers":

                last_year = ax.bar(x - width/2, np.array(ly)-np.array(lyp), width, color="goldenrod", 
                                bottom=lyp, label=season+" (Actual)")
                last_year_pred = ax.bar(x - width/2, lyp, width, alpha=0.7,
                                        color="seagreen", label=season+" (Predicted)")
                
                correctness = [val>0 for val in (np.array(ly)-np.array(ty))]
            
            elif perf_type=="Underperformers":

                last_year = ax.bar(x - width/2, ly, width, color="goldenrod", 
                                label=season+" (Actual)")
                last_year_pred = ax.bar(x - width/2, np.array(lyp)-np.array(ly), 
                                        width, bottom=ly, color="seagreen",
                                        alpha=0.7, label=season+" (Predicted)")
                
                correctness = [val<0 for val in (np.array(ly)-np.array(ty))]
                
            this_year = ax.bar(x + width/2, ty, width, 
                            color="powderblue", label=future_season+" (Actual)")

            ax.set_ylabel('Goal Pace over 82 Games')
            
            plt.title("Top "+str(np.max([top_n,10]))+" "+perf_type+" of "+season)
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            for i in range(len(x)):
                ax.get_xticklabels()[i].set_color(CorrectColor(correctness[i])) 
            ax.legend()

            def labeler(graph):
                for bars in graph:
                    height = bars.get_height()
                    ax.annotate('{}'.format(round(height,1)),
                                xy=(bars.get_x() + bars.get_width() / 2, height/2),
                                xytext=(0,0),
                                textcoords="offset points",
                                ha='center', va='bottom')
            if perf_type=="Overperformers":
                for bars in zip(last_year, last_year_pred):
                    height_ly = bars[0].get_height()
                    height_lyp = bars[1].get_height()
                    ax.annotate('{}'.format(round(height_ly+height_lyp,1)),
                                xy=(bars[0].get_x() + bars[0].get_width() / 2, height_ly/2+height_lyp),
                                xytext=(0,0),
                                textcoords="offset points",
                                ha='center', va='bottom')
                labeler(last_year_pred)

            elif perf_type=="Underperformers":
                for bars in zip(last_year, last_year_pred):
                    height_ly = bars[0].get_height()
                    height_lyp = bars[1].get_height()
                    ax.annotate('{}'.format(round(height_ly+height_lyp,1)),
                                xy=(bars[0].get_x() + bars[0].get_width() / 2, height_ly+height_lyp/2),
                                xytext=(0,0),
                                textcoords="offset points",
                                ha='center', va='bottom')
                labeler(last_year)
            
            labeler(this_year)

            fig.set_size_inches(17, 5.5)
            fig.tight_layout()

            plt.show()