
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")


class StudentAgent(Agent):

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"
    self.score = 0
    self.static_score_changes = []
    self.transpositionTable = {}

  weight_matrices = {
  6: [[5,-3,2,2,-3,5],
      [-3,-4,-1,-1,-4,-3],
      [2,-1,1,1,-1,2],
      [2,-1,1,1,-1,2],
      [-3,-4,-1,-1,-4,-3],
      [5,-3,2,2,2,-3,5]],
    
  8:  [[5, -3, 2, 2, 2, 2, -3, 5],
       [-3, -4, -1, -1, -1, -1, -4, -3],
       [2, -1, 1, 0, 0, 1, -1, 2],
       [2, -1, 0, 1, 1, 0, -1, 2],
       [2, -1, 0, 1, 1, 0, -1, 2],
       [2, -1, 1, 0, 0, 1, -1, 2],
       [-3, -4, -1, -1, -1, -1, -4, -3],
       [5, -3, 2, 2, 2, 2, -3, 5]],
       
  10:  
       [[5, -3,  2,  2,  1,  1,  2,  2, -3,  5],
       [-3, -4, -1, -1, -1, -1, -1, -1, -4, -3],
       [2, -1,  1,  0,  0,  0,  0,  1, -1,  2],
       [2, -1,  0,  1,  1,  1,  1,  0, -1,  2],
       [1, -1,  0,  1,  1,  1,  1,  0, -1,  1],
       [1, -1,  0,  1,  1,  1,  1,  0, -1,  1],
       [2, -1,  0,  1,  1,  1,  1,  0, -1,  2],
       [2, -1,  1,  0,  0,  0,  0,  1, -1,  2],
       [-3, -4, -1, -1, -1, -1, -1, -1, -4, -3],
       [5, -3,  2,  2,  1,  1,  2,  2, -3,  5]],
  
  12:  
       [[5, -3,  2,  2,  2,  1,  1,  2,  2,  2, -3,  5],
       [-3, -4, -1, -1, -1, -1, -1, -1, -1, -1, -4, -3],
       [2, -1,  1,  0,  0,  0,  0,  0,  0,  1, -1,  2],
       [2, -1,  0,  1,  1,  1,  1,  1,  1,  0, -1,  2],
       [2, -1,  0,  1,  1,  1,  1,  1,  1,  0, -1,  2],
       [1, -1,  0,  1,  1,  1,  1,  1,  1,  0, -1,  1],
       [1, -1,  0,  1,  1,  1,  1,  1,  1,  0, -1,  1],
       [2, -1,  0,  1,  1,  1,  1,  1,  1,  0, -1,  2],
       [2, -1,  0,  1,  1,  1,  1,  1,  1,  0, -1,  2],
       [2, -1,  1,  0,  0,  0,  0,  0,  0,  1, -1,  2],
       [-3, -4, -1, -1, -1, -1, -1, -1, -1, -1, -4, -3],
       [5, -3,  2,  2,  2,  1,  1,  2,  2,  2, -3,  5]]
  
  }

  def corner_heuristic(self, chess_board, player):

    if player == 1:
      opponent = 2
    else:
      opponent = 1

    board_size = chess_board.shape[0] -1
    
    corners = [(0,0), (0,board_size), (board_size, 0), (board_size, board_size)]
    
    player_corners = sum(chess_board[i,j] == player for i,j in corners)
    opponent_corners = sum(chess_board[i,j] == opponent for i,j in corners)
    if (player_corners + opponent_corners == 0):
      return 0
    
    percent = 100*(player_corners - opponent_corners)/(player_corners + opponent_corners)

    return percent
    # add potential corners (?)
  
  def num_pieces_heuristic(self, chess_board, player):
    
    if player == 1:
      opponent = 2
    else:
      opponent = 1

    player_pieces = np.sum(chess_board == player)
    opponent_pieces = np.sum(chess_board == opponent)
    
    total_coins = player_pieces + opponent_pieces
    
    percent = (player_pieces - opponent_pieces)/ (player_pieces + opponent_pieces) * 100
    
    return percent, total_coins
    
  def mobility_heuristic(self, chess_board, player):
    if player == 1:
      opponent = 2
    else:
      opponent = 1
    
    board_size = chess_board.shape[0] -1
    
    corners = [(0,0), (0,board_size), (board_size, 0), (board_size, board_size)]
    
    player_moves = get_valid_moves(chess_board, player)
    opponent_moves = get_valid_moves(chess_board, opponent)

    len_player_moves = len(player_moves)
    len_opponent_moves = len(opponent_moves)

    if (len_player_moves + len_opponent_moves) != 0:
      mobility_h = (len_player_moves - len_opponent_moves)/ (len_player_moves + len_opponent_moves)*100
    else:
      mobility_h = 0
    

    return (mobility_h)
    # add future mobility ?
  
  def eval_board(self,chess_board, player, num_tiles):
    # add something to priotitize edges ?
    corners = self.corner_heuristic(chess_board, player)
    
    mobility = self.mobility_heuristic(chess_board, player)
    board_size = chess_board.shape[0] -1
    if board_size +1 == 6:
      return  ( 0.4*mobility + 0.6*corners )
    else:
      return ( 0.3*mobility + 0.7*corners )
  
  def alpha_beta_prune(self, chess_board, player, opponent,
                        isMaxPlayer, depth, alpha, beta, num_tiles, start_time):
    
    hashcode = hash(tuple(map(tuple, chess_board)))
    if hashcode in self.transpositionTable and self.transpositionTable[hashcode] is not None:
        entry = self.transpositionTable[hashcode]
        
        if (entry["depth"] >= depth):
            
            if (entry["type"] == "exact"):
                return entry["score"], entry["move"]
            elif (entry["type"] == "lower"):
                alpha = max(alpha,entry["score"])
            elif (entry["type"] == "upper"):
                
                beta = min(beta,entry["score"])
            if alpha >= beta:
                
                return entry["score"], entry["move"] #problematic?
          
        
    board_size = chess_board.shape[0]
    

    if time.time() - start_time > 1.9 or depth == 0 or check_endgame(chess_board,player,opponent)[0] == True:
      
      result = self.eval_board(chess_board, player, num_tiles)
      #self.transpositionTable[hashcode] = {"score": result, "depth": depth, "type": "exact", "move": None}
      return result, None

    if isMaxPlayer:
      
      valid_moves = sorted(get_valid_moves(chess_board, player),
                           key=lambda move: 
                           self.weight_matrices[board_size][move[0]][move[1]], 
                           reverse=True)
      if not valid_moves:
        # evaluation of board
        return self.eval_board(chess_board, player, num_tiles), None
      
      max_eval = (float('-inf'),None)

      for move in valid_moves:
        new_board = chess_board.copy()
        execute_move(new_board, move, player)

        eval = self.alpha_beta_prune(new_board, player, opponent,
                        False, depth -1, alpha, beta, num_tiles +1, start_time)

        if eval[0] > max_eval[0]:
          max_eval = (eval[0], move)
      
        alpha = max(alpha, max_eval[0])

        if beta <= alpha:
          break
    
      if max_eval[0] >= beta:
        
        self.transpositionTable[hashcode] = {"score": max_eval[0], "depth": depth, "type": "lower", "move": max_eval[1]}
      elif max_eval[0] <= alpha:
      
        self.transpositionTable[hashcode] = {"score": max_eval[0], "depth": depth, "type": "upper", "move": max_eval[1]}
      else:
        self.transpositionTable[hashcode] = {"score": max_eval[0], "depth": depth, "type": "exact", "move": max_eval[1]}
     
      
      return max_eval
    
    else:

      valid_moves = sorted(get_valid_moves(chess_board, opponent),
                           key=lambda move: 
                           self.weight_matrices[board_size][move[0]][move[1]], 
                           reverse=True)
      if not valid_moves:
        # evaluation of board
        return self.eval_board(chess_board, opponent, num_tiles), None
      
      min_eval = (float('inf'), None)

      for move in valid_moves:
        new_board = chess_board.copy()
        execute_move(new_board, move, opponent)

        eval = self.alpha_beta_prune(new_board, player, opponent,
                        True, depth -1, alpha, beta, num_tiles +1, start_time)
        

        if eval[0] < min_eval[0]:
          min_eval = (eval[0], move)
        
        beta = min(beta, min_eval[0])

        if beta <= alpha:
          break
      

      if min_eval[0] >= beta:
    
        self.transpositionTable[hashcode] = {"score": min_eval[0], "depth": depth, "type": "lower", "move": min_eval[1]}
      elif min_eval[0] <= alpha:
       
        self.transpositionTable[hashcode] = {"score": min_eval[0], "depth": depth, "type": "upper", "move": min_eval[1]}
      else:
        self.transpositionTable[hashcode] = {"score": min_eval[0], "depth": depth, "type": "exact", "move": min_eval[1]}
       
      return min_eval

        
  def step(self, chess_board, player, opponent):
    self.transpositionTable = {}

    start_time = time.time()
    
    # alpha-beta pruning (bounded-depth) with heuristic calculations at the end-points
    
    isMaxPlayer = True
    depth = 1
    alpha = float('-inf')
    beta = float('inf')

    best_move_so_far = None

    while True:

      alpha = float('-inf')
      beta = float('inf')
      isMaxPlayer = True

      if time.time() - start_time > 1.9:
        break

      score, poss_move = self.alpha_beta_prune(chess_board, player, opponent, isMaxPlayer, depth, alpha, beta, 4, start_time)

      if time.time() - start_time < 1.9 and poss_move != None:
        best_move_so_far = poss_move

      depth += 1


    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")
  
    return best_move_so_far
