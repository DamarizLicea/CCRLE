def q_rl_agent(self, alpha=0.2, gamma=0.9, epsilon=0.9, min_epsilon=0.01, decay_rate=0.99, max_steps=70, episodes=5):
        """Entrena a un agente usando q-learning normal y epsilon greedy ."""
        min_steps_to_goal = float('inf')
        best_episode = None
        successful_episodes = 0
        
        self.initialize_q_table(2)
        
        for episode in range(episodes):
            current_pos = self.rl_agent_start_pos
            steps = 0
            rewards_collected = 0
            total_reward = 0 
            
            self._gen_grid(self.grid.width, self.grid.height)
            current_quadrant = self.quadrants.index(self.current_quadrant)

            
            while steps < max_steps and rewards_collected < 1:
                state = self.encode_state(*current_pos) + current_quadrant

                if state not in self.q_table_rl_agent:
                    self.q_table_rl_agent[state] = {0: 0, 1: 0, 2: 0, 3: 0}

                if random.uniform(0, 1) < epsilon:
                    action = random.choice(list(self.q_table_rl_agent[state].keys()))
                else:
                    action = max(self.q_table_rl_agent[state], key=self.q_table_rl_agent[state].get)

                new_pos = self.next_position(current_pos, action)
                new_state = self.encode_state(*new_pos) + current_quadrant

                if new_state not in self.q_table_rl_agent:
                    self.q_table_rl_agent[new_state] = {0: 0, 1: 0, 2: 0, 3: 0}

                reward = -1

                if new_pos in self.reward_positions:
                    reward += 30 
                    print(f"Recompensa recogida en {new_pos}")
                    rewards_collected += 1
                    self.reward_positions.remove(new_pos)
                    self.grid.set(*new_pos, None)

                    if rewards_collected == 1:
                        successful_episodes += 1 
                        total_reward += reward

                        if steps + 1 < min_steps_to_goal:
                            min_steps_to_goal = steps + 1
                            best_episode2 = episode + 1
                            break

                total_reward += reward

                self.q_table_rl_agent[state][action] += alpha * (reward + gamma * max(self.q_table_rl_agent[new_state].values()) - self.q_table_rl_agent[state][action])
                current_pos = new_pos
                self.agent_pos = current_pos
                epsilon = max(min_epsilon, epsilon * decay_rate)

                self.render()
                steps += 1
                
                if steps >= max_steps:
                    break


        print(f"\nEntrenamiento completado con {successful_episodes} episodios exitosos de {episodes}")
        self.save_q_table(self.q_table_rl_agent, "q_table21.txt")
        if rewards_collected == 1:
                done = True
                print(f"Se lograron recoger las 1 recompensas en el episodio {episode}.")
        else:
                print("No se lograron recoger las 1 recompensas en ningún episodio.")
        return rewards_collected