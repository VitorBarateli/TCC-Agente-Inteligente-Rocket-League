from typing import List, Dict, Any
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values
import numpy as np

class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    """Recompensa o agente por se mover rapidamente em direção à bola."""
    # Recompensa entre 0.1 e 1
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_physics = car.physics if car.is_orange else car.inverted_physics
            ball_physics = state.ball if car.is_orange else state.inverted_ball
            player_vel = car_physics.linear_velocity
            pos_diff = (ball_physics.position - car_physics.position)
            dist_to_ball = np.linalg.norm(pos_diff)
            dir_to_ball = pos_diff / dist_to_ball

            speed_toward_ball = np.dot(player_vel, dir_to_ball)

            rewards[agent] = max(speed_toward_ball / common_values.CAR_MAX_SPEED, 0.0)

        return rewards

class InAirReward(RewardFunction[AgentID, GameState, float]):
    """Recompensa o agente por estar no ar"""
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: float(not state.cars[agent].on_ground) for agent in agents}

class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """Recompensa o jogador por acertar a bola em direção ao gol adversário"""
    # Recompensa entre 0.001 e 0.5
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            if car.is_orange:
                goal_y = -common_values.BACK_NET_Y
            else:
                goal_y = common_values.BACK_NET_Y

            ball_vel = ball.linear_velocity
            pos_diff = np.array([0, goal_y, 0]) - ball.position
            dist = np.linalg.norm(pos_diff)
            dir_to_goal = pos_diff / dist
            
            vel_toward_goal = np.dot(ball_vel, dir_to_goal)
            rewards[agent] = max(vel_toward_goal / common_values.BALL_MAX_SPEED, 0)

        return rewards

class FaceBallReward(RewardFunction[AgentID, GameState, float]):
    """Recompensa o agente por estar de frente para a bola"""
    # Recompensa entre -1 e 1
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_physics = car.physics if car.is_orange else car.inverted_physics
            ball_physics = state.ball if car.is_orange else state.inverted_ball

            pos_diff = ball_physics.position - car_physics.position
            norm = np.linalg.norm(pos_diff)
            if norm == 0:
                rewards[agent] = 0.0
                continue

            norm_pos_diff = pos_diff / norm
            forward = car_physics.forward
            facing_reward = np.dot(forward, norm_pos_diff)
            rewards[agent] = float(facing_reward)

        return rewards

class BallCloserToGoalReward(RewardFunction[AgentID, GameState, float]):
    """Recompensa se a bola estiver mais próxima do gol adversário em relação ao passo anterior"""
    # Recompensa por Chutar entre -160 e 160

    def reset(self, agents, initial_state, shared_info):
        self.last_dists = {agent: None for agent in agents}

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            if car.is_orange:
                goal_y = -common_values.BACK_NET_Y
            else:
                goal_y = common_values.BACK_NET_Y

            goal_pos = np.array([0, goal_y, 0])
            ball_pos = state.ball.position
            dist = np.linalg.norm(goal_pos - ball_pos)

            last = self.last_dists.get(agent)
            reward = 0.0
            if last is not None:
                reward = last - dist  # recompensa positiva se a distância diminuiu

            self.last_dists[agent] = dist
            rewards[agent] = reward

        return rewards

class FlipReward(RewardFunction[AgentID, GameState, float]):
    """Recompensa o agente por usar flip de forma estratégica: após tocar na bola ou sem boost"""

    def reset(self, agents, initial_state, shared_info):
        self.last_flips = {agent: False for agent in agents}
        self.last_touches = {agent: 0 for agent in agents}

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            flipped_now = car.has_flipped and not self.last_flips.get(agent, False)

            # Checa se tocou na bola neste tick
            touched_now = car.ball_touches > self.last_touches.get(agent, 0)
            low_boost = car.boost_amount < 5  # ou outro limiar

            reward = 0.0
            if flipped_now and (touched_now or low_boost):
                reward = 1.0  # ajustável

            # Atualiza estado anterior
            self.last_flips[agent] = car.has_flipped
            self.last_touches[agent] = car.ball_touches
            rewards[agent] = reward

        return rewards

class UsefulFlipReward(RewardFunction[AgentID, GameState, float]):
    """
    Recompensa proporcional ao quão útil foi o flip:
    - Para chutar a bola (sempre recompensa chute, força + bônus se for pro gol).
    - Para ganhar velocidade de movimento.
    - Para ganhar velocidade na direção da bola.
    - Considera flips feitos até X ticks antes do toque como chute.
    """

    def __init__(
        self,
        ball_scale: float = 1,
        car_scale: float = 1,
        toward_ball_bonus: float = 1.66,
        goal_bonus: float = 3.3,
        max_flip_delay_ticks: int = 25
    ):
        """
        :param ball_scale: fator multiplicador para ganho de velocidade da bola (normalizado).
        :param car_scale: fator multiplicador para ganho de velocidade do carro (normalizado).
        :param toward_ball_bonus: fator multiplicador extra quando o ganho de velocidade for na direção da bola.
        :param goal_bonus: multiplicador extra se o chute for na direção do gol.
        :param max_flip_delay_ticks: ticks máximos entre o flip e o toque para considerar como chute.
        """
        self.ball_scale = ball_scale
        self.car_scale = car_scale
        self.toward_ball_bonus = toward_ball_bonus
        self.goal_bonus = goal_bonus
        self.max_flip_delay_ticks = max_flip_delay_ticks

    def reset(self, agents, initial_state, shared_info):
        self.last_flips = {agent: False for agent in agents}
        self.last_flip_tick = {agent: -999 for agent in agents}
        self.last_ball_vel = initial_state.ball.linear_velocity.copy()
        self.last_car_vel = {
            agent: initial_state.cars[agent].physics.linear_velocity.copy()
            for agent in agents
        }
        self.last_touches = {agent: 0 for agent in agents}
        self.tick_count = 0

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        current_ball_vel = state.ball.linear_velocity
        self.tick_count += 1

        for agent in agents:
            car = state.cars[agent]
            flipped_now = car.has_flipped and not self.last_flips.get(agent, False)
            touched_now = car.ball_touches > self.last_touches.get(agent, 0)
            reward = 0.0

            # Registra quando o flip foi feito
            if flipped_now:
                self.last_flip_tick[agent] = self.tick_count

            # ----- Chute na bola -----
            if touched_now:
                ticks_since_flip = self.tick_count - self.last_flip_tick[agent]
                if 0 <= ticks_since_flip <= self.max_flip_delay_ticks:
                    # Posição do gol adversário
                    if car.is_orange:
                        goal_y = -common_values.BACK_NET_Y
                    else:
                        goal_y = common_values.BACK_NET_Y

                    goal_pos = np.array([0, goal_y, 0])
                    dir_to_goal = goal_pos - state.ball.position
                    dist_to_goal = np.linalg.norm(dir_to_goal)
                    if dist_to_goal > 0:
                        dir_to_goal /= dist_to_goal

                    # Força do chute = aumento total da velocidade da bola
                    ball_speed_change = max(
                        0.0,
                        np.linalg.norm(current_ball_vel) - np.linalg.norm(self.last_ball_vel)
                    )

                    # Recompensa base
                    chute_reward = (ball_speed_change / common_values.BALL_MAX_SPEED) * self.ball_scale

                    # Bônus se a bola for para o gol
                    if np.linalg.norm(current_ball_vel) > 1e-6:
                        ball_dir = current_ball_vel / np.linalg.norm(current_ball_vel)
                        alignment = np.dot(ball_dir, dir_to_goal)
                        if alignment > 0:
                            chute_reward *= (1.0 + alignment * (self.goal_bonus - 1.0))

                    reward += chute_reward
                    #print(f"Agente {agent}: Recompensa por chute = {chute_reward:.2f}, Mudança na velocidade da bola = {ball_speed_change:.2f}")

            # ----- Ganho de velocidade do carro -----
            elif flipped_now:
                prev_speed = np.linalg.norm(self.last_car_vel[agent])
                new_speed = np.linalg.norm(car.physics.linear_velocity)
                speed_gain = max(0.0, new_speed - prev_speed)

                car_speed_reward = (speed_gain / common_values.CAR_MAX_SPEED) * self.car_scale

                # Bônus por alinhar velocidade com a direção da bola
                car_pos = car.physics.position if not car.is_orange else car.inverted_physics.position
                ball_pos = state.ball.position if not car.is_orange else state.inverted_ball.position
                dir_to_ball = ball_pos - car_pos
                dist_to_ball = np.linalg.norm(dir_to_ball)
                if dist_to_ball > 0:
                    dir_to_ball /= dist_to_ball
                    vel_dir = car.physics.linear_velocity / (np.linalg.norm(car.physics.linear_velocity) + 1e-6)
                    alignment = max(0.0, np.dot(vel_dir, dir_to_ball))
                    car_speed_reward *= (1 + alignment * self.toward_ball_bonus)

                reward += car_speed_reward

            # Atualiza históricos
            self.last_flips[agent] = car.has_flipped
            self.last_touches[agent] = car.ball_touches
            self.last_car_vel[agent] = car.physics.linear_velocity

            rewards[agent] = reward

        self.last_ball_vel = current_ball_vel
        return rewards

class AdvancedUsefulFlipReward(RewardFunction[AgentID, GameState, float]):
    """
    Recompensa proporcional ao quão útil foi o flip:
    - Para chutar a bola (sempre recompensa chute, força + bônus se for pro gol).
    - Para ganhar velocidade de movimento.
    - Para ganhar velocidade na direção da bola.
    - Considera flips feitos até X ticks antes do toque como chute.
    - Novo multiplicador: recompensa extra se o oponente estiver próximo da bola.
    """
    # Recompensa por Chutar entre 0.1 e 4
    # Recompensa por Flipar entre 0.2 e 0.6

    def __init__(
        self,
        ball_scale: float = 2,
        car_scale: float = 1,
        toward_ball_bonus: float = 1.66,
        goal_bonus: float = 1.66,
        max_flip_delay_ticks: int = 25,
        opponent_distance_multiplier: float = 2.0  # Novo multiplicador baseado na distância do oponente
    ):
        self.ball_scale = ball_scale
        self.car_scale = car_scale
        self.toward_ball_bonus = toward_ball_bonus
        self.goal_bonus = goal_bonus
        self.max_flip_delay_ticks = max_flip_delay_ticks
        self.opponent_distance_multiplier = opponent_distance_multiplier

    def reset(self, agents, initial_state, shared_info):
        self.last_flips = {agent: False for agent in agents}
        self.last_flip_tick = {agent: -999 for agent in agents}
        self.last_ball_vel = initial_state.ball.linear_velocity.copy()
        self.last_car_vel = {
            agent: initial_state.cars[agent].physics.linear_velocity.copy()
            for agent in agents
        }
        self.last_touches = {agent: 0 for agent in agents}
        self.tick_count = 0

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        current_ball_vel = state.ball.linear_velocity
        self.tick_count += 1

        for agent in agents:
            car = state.cars[agent]
            flipped_now = car.has_flipped and not self.last_flips.get(agent, False)
            touched_now = car.ball_touches > self.last_touches.get(agent, 0)
            reward = 0.0

            # Registra quando o flip foi feito
            if flipped_now:
                self.last_flip_tick[agent] = self.tick_count

            # ----- Chute na bola -----
            if touched_now:
                ticks_since_flip = self.tick_count - self.last_flip_tick[agent]
                if 0 <= ticks_since_flip <= self.max_flip_delay_ticks:
                    # Posição do gol adversário
                    if car.is_orange:
                        goal_y = -common_values.BACK_NET_Y
                    else:
                        goal_y = common_values.BACK_NET_Y

                    goal_pos = np.array([0, goal_y, 0])
                    dir_to_goal = goal_pos - state.ball.position
                    dist_to_goal = np.linalg.norm(dir_to_goal)
                    if dist_to_goal > 0:
                        dir_to_goal /= dist_to_goal

                    # Força do chute = aumento da velocidade da bola
                    ball_speed_change = max(
                        0.0,
                        np.linalg.norm(current_ball_vel) - np.linalg.norm(self.last_ball_vel)
                    )

                    chute_reward = (ball_speed_change / common_values.BALL_MAX_SPEED) * self.ball_scale

                    # Bônus se a bola for para o gol
                    if np.linalg.norm(current_ball_vel) > 1e-6:
                        ball_dir = current_ball_vel / np.linalg.norm(current_ball_vel)
                        alignment = np.dot(ball_dir, dir_to_goal)
                        if alignment > 0:
                            chute_reward *= (1.0 + alignment * (self.goal_bonus - 1.0))

                    # ----- Novo multiplicador baseado na proximidade do oponente -----
                    opponents = [c for i, c in state.cars.items() if i != agent]
                    if opponents:
                        opponent = min(
                            opponents,
                            key=lambda o: np.linalg.norm(
                                (o.inverted_physics.position if car.is_orange else o.physics.position) - state.ball.position
                            )
                        )
                        opp_pos = opponent.inverted_physics.position if car.is_orange else opponent.physics.position
                        opponent_dist = np.linalg.norm(opp_pos - state.ball.position)

                        # Normaliza pela largura total do campo (2 * SIDE_WALL_X = 8192 uu)
                        max_field_dist = 2 * common_values.SIDE_WALL_X
                        opponent_multiplier = max(0.0, 1 - opponent_dist / max_field_dist) * self.opponent_distance_multiplier

                        chute_reward *= (1 + opponent_multiplier)

                    reward += chute_reward

            # ----- Ganho de velocidade do carro -----
            elif flipped_now:
                prev_speed = np.linalg.norm(self.last_car_vel[agent])
                new_speed = np.linalg.norm(car.physics.linear_velocity)
                speed_gain = max(0.0, new_speed - prev_speed)

                car_speed_reward = (speed_gain / common_values.CAR_MAX_SPEED) * self.car_scale

                # Bônus por alinhar velocidade com a direção da bola
                car_pos = car.physics.position if not car.is_orange else car.inverted_physics.position
                ball_pos = state.ball.position if not car.is_orange else state.inverted_ball.position
                dir_to_ball = ball_pos - car_pos
                dist_to_ball = np.linalg.norm(dir_to_ball)
                if dist_to_ball > 0:
                    dir_to_ball /= dist_to_ball
                    vel_dir = car.physics.linear_velocity / (np.linalg.norm(car.physics.linear_velocity) + 1e-6)
                    alignment = max(0.0, np.dot(vel_dir, dir_to_ball))
                    car_speed_reward *= (1 + alignment * self.toward_ball_bonus)

                reward += car_speed_reward

            # Atualiza históricos
            self.last_flips[agent] = car.has_flipped
            self.last_touches[agent] = car.ball_touches
            self.last_car_vel[agent] = car.physics.linear_velocity

            rewards[agent] = reward

        self.last_ball_vel = current_ball_vel
        return rewards

class TouchForceReward(RewardFunction[AgentID, GameState, float]):
    """
    Recompensa proporcional à mudança de velocidade da bola causada por um toque do agente.
    Toques fracos dão pouca recompensa, toques fortes dão mais.
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_ball_vel = initial_state.ball.linear_velocity.copy()
        self.last_touches = {agent: 0 for agent in agents}

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:

        rewards = {}
        current_vel = state.ball.linear_velocity

        for agent in agents:
            car = state.cars[agent]
            touched = car.ball_touches > self.last_touches.get(agent, 0)

            reward = 0.0
            if touched:
                vel_change = np.linalg.norm(current_vel - self.last_ball_vel)
                reward = vel_change / common_values.BALL_MAX_SPEED  # Normaliza para [0, 1]

            self.last_touches[agent] = car.ball_touches
            rewards[agent] = reward

        self.last_ball_vel = current_vel
        return rewards

class AdvancedTouchForceReward(RewardFunction[AgentID, GameState, float]):
    """
    Recompensa proporcional à mudança de velocidade da bola causada por um toque do agente,
    MAS apenas se o toque não tiver sido feito com um flip.
    Ou seja, recompensa empurrões/carregadas da bola.
    """
    # Recompensa entre 0.1 e 0.4

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_ball_vel = initial_state.ball.linear_velocity.copy()
        self.last_touches = {agent: 0 for agent in agents}
        self.last_flips = {agent: initial_state.cars[agent].has_flipped for agent in agents}

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:

        rewards = {}
        current_vel = state.ball.linear_velocity

        for agent in agents:
            car = state.cars[agent]
            touched = car.ball_touches > self.last_touches.get(agent, 0)

            reward = 0.0
            if touched:
                # Só recompensa se o agente NÃO estiver usando flip
                if not car.has_flipped:
                    vel_change = np.linalg.norm(current_vel - self.last_ball_vel)
                    reward = vel_change / common_values.BALL_MAX_SPEED  # Normaliza em [0,1]

            # Atualiza histórico
            self.last_touches[agent] = car.ball_touches
            self.last_flips[agent] = car.has_flipped
            rewards[agent] = reward

        self.last_ball_vel = current_vel
        return rewards
    
class CollectBoostReward(RewardFunction[AgentID, GameState, float]):
    """
    Recompensa o agente por coletar boost (quando a quantidade de boost aumenta).
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_boost = {agent: initial_state.cars[agent].boost_amount for agent in agents}

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            current_boost = car.boost_amount
            last_boost = self.last_boost.get(agent, current_boost)

            reward = 0.0
            if current_boost > last_boost:
                # Recompensa proporcional ao boost coletado (pode ser ajustado)
                reward = (current_boost - last_boost) / 100.0

            self.last_boost[agent] = current_boost
            rewards[agent] = reward

        return rewards
    
class AdvancedCollectBoostReward(RewardFunction[AgentID, GameState, float]):
    """
    Recompensa o agente por coletar boost (quando a quantidade de boost aumenta).
    Quanto mais distante o oponente estiver da bola, maior será a recompensa.
    Mas NÃO recompensa durante o kickoff.
    """
    # Recompensa entre 0.1 e 2

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_boost = {agent: initial_state.cars[agent].boost_amount for agent in agents}

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        ball_pos = state.ball.position

        # 🚫 Bloqueia recompensa se ainda estamos no kickoff
        if shared_info.get("kickoff_active", False):
            return {agent: 0.0 for agent in agents}

        for agent in agents:
            car = state.cars[agent]
            current_boost = car.boost_amount
            last_boost = self.last_boost.get(agent, current_boost)

            reward = 0.0
            if current_boost > last_boost:
                # Recompensa base proporcional ao boost coletado
                base_reward = (current_boost - last_boost) / 100.0

                # Distância do oponente mais próximo até a bola
                opponents = [c for i, c in state.cars.items() if i != agent]
                if opponents:
                    opponent = min(
                        opponents,
                        key=lambda o: np.linalg.norm(o.physics.position - ball_pos)
                    )
                    opponent_dist = np.linalg.norm(opponent.physics.position - ball_pos)

                    # Normaliza pela largura do campo (2 * SIDE_WALL_X)
                    max_field_dist = 2 * common_values.SIDE_WALL_X
                    dist_factor = opponent_dist / max_field_dist  # valor ∈ [0,1]

                    reward = base_reward * (1 + dist_factor)
                else:
                    reward = base_reward
                

            self.last_boost[agent] = current_boost
            rewards[agent] = reward

        return rewards

class SaveBoostReward(RewardFunction[AgentID, GameState, float]):
    """
    Recompensa o agente por conservar boost. Calcula sqrt(boost_amount), normalizado para [0, 1].
    """
    # Recompensa entre 0.1 e 1

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass  # Nenhum estado anterior necessário

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}

        if shared_info.get("kickoff_active", False):
            return {agent: 0.0 for agent in agents}

        for agent in agents:
            car = state.cars[agent]
            # sqrt(boost) / sqrt(100) normaliza o valor para o intervalo [0, 1]
            reward = np.sqrt(car.boost_amount) / np.sqrt(100.0)
            rewards[agent] = reward
        return rewards
    
class WasteBoostAtMaxSpeedPunishment(RewardFunction[AgentID, GameState, float]):
    """
    Punição quando o agente usa boost enquanto já está na velocidade máxima.
    """
    # Recompensa -1 ou 0

    def reset(self, agents, initial_state, shared_info):
        pass  # Sem histórico necessário

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            vel = np.linalg.norm(car.physics.linear_velocity)

            # Só considera velocidade máxima mesmo
            at_max_speed = vel >= 0.999 * common_values.CAR_MAX_SPEED
            is_wasting_boost = car.is_boosting and at_max_speed

            reward = -1.0 if is_wasting_boost else 0.0
            rewards[agent] = reward
        return rewards

class AerialTouchReward(RewardFunction[AgentID, GameState, float]):
    """
    Recompensa o agente por tocar na bola no ar.
    Só conta se a bola também estiver acima de uma altura mínima.
    A recompensa é baseada na fração do tempo no ar e na altura da bola.
    """
    # Recompensa entre 0.07 e 0.2

    MAX_TIME_IN_AIR = 1.75  # estimativa do tempo máximo razoável de voo

    def __init__(self, min_ball_height: float = 150.0):
        """
        :param min_ball_height: altura mínima (em uu) da bola para contar como aerial.
                                Exemplo: ~150 ≈ metade da altura do carro.
        """
        self.min_ball_height = min_ball_height

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_touches = {agent: 0 for agent in agents}

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:

        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            touched_now = car.ball_touches > self.last_touches.get(agent, 0)

            reward = 0.0
            # agora exige: carro no ar + bola acima do mínimo configurado
            if touched_now and not car.on_ground and state.ball.position[2] > self.min_ball_height:
                air_time_frac = min(car.air_time_since_jump, self.MAX_TIME_IN_AIR) / self.MAX_TIME_IN_AIR
                height_frac = state.ball.position[2] / common_values.CEILING_Z
                reward = min(air_time_frac, height_frac)

            self.last_touches[agent] = car.ball_touches
            rewards[agent] = reward

        return rewards

    
class KickoffReward(RewardFunction[AgentID, GameState, float]):
    """
    Recompensa durante o kickoff para incentivar chegar o mais rápido possível na bola:
    - Velocidade em direção à bola (quanto mais rápido, melhor).
    - Progresso em direção à bola.
    - Bônus alto no primeiro toque.
    - Penalidade indireta: a recompensa decai com o tempo.
    """

    def reset(self, agents, initial_state, shared_info):
        self.start_dist = {}
        self.kickoff_active = True
        self.first_touch = {agent: False for agent in agents}
        self.tick_count = 0

        ball_pos = initial_state.ball.position
        for agent in agents:
            car_pos = initial_state.cars[agent].physics.position
            self.start_dist[agent] = np.linalg.norm(ball_pos - car_pos)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        if not self.kickoff_active:
            return {agent: 0.0 for agent in agents}

        self.tick_count += 1
        ball_pos = state.ball.position
        rewards = {}

        for agent in agents:
            car = state.cars[agent]
            car_pos = car.physics.position
            car_vel = car.physics.linear_velocity

            # ----- Progresso até a bola -----
            dist_now = np.linalg.norm(ball_pos - car_pos)
            progress = (self.start_dist[agent] - dist_now) / self.start_dist[agent]
            progress = max(progress, 0.0)

            # ----- Velocidade em direção à bola -----
            if dist_now > 1e-6:
                dir_to_ball = (ball_pos - car_pos) / dist_now
                speed_toward_ball = np.dot(car_vel, dir_to_ball)
                speed_reward = max(speed_toward_ball / common_values.CAR_MAX_SPEED, 0.0)
            else:
                speed_reward = 0.0

            # ----- Decaimento com o tempo -----
            time_decay = max(0.1, 1 - self.tick_count / 200)  
            # depois de ~200 ticks (≈3s), o valor cai bastante

            reward = (progress + speed_reward) * time_decay

            # ----- Primeiro toque -----
            if car.ball_touches > 0 and not self.first_touch[agent]:
                self.first_touch[agent] = True
                self.kickoff_active = False  # kickoff terminou
                reward += 5.0  # bônus grande pelo primeiro toque

            rewards[agent] = reward

        return rewards

class AdvancedKickoffReward(RewardFunction[AgentID, GameState, float]):
    """
    Recompensa durante o kickoff para incentivar chegar o mais rápido possível na bola:
    - Velocidade em direção à bola (quanto mais rápido, melhor).
    - Progresso em direção à bola.
    - Alinhamento do carro em linha reta até a bola (sem tremer/desviar).
    - Penalidade forte para desvios laterais.
    - Recompensa extra pelo uso de boost durante o kickoff.
    - Bônus no primeiro toque (justo para todos que tocarem no mesmo tick).
    - Penalidade indireta: a recompensa decai com o tempo.
    """

    def reset(self, agents, initial_state, shared_info):
        self.start_dist = {}
        self.start_positions = {}
        self.kickoff_active = True
        self.first_touch = {agent: False for agent in agents}
        self.tick_count = 0

        ball_pos = initial_state.ball.position
        for agent in agents:
            car_pos = initial_state.cars[agent].physics.position
            self.start_dist[agent] = np.linalg.norm(ball_pos - car_pos)
            self.start_positions[agent] = car_pos.copy()

        # Marca kickoff ativo no início
        shared_info["kickoff_active"] = True

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        if not self.kickoff_active:
            shared_info["kickoff_active"] = False
            return {agent: 0.0 for agent in agents}

        shared_info["kickoff_active"] = True
        self.tick_count += 1
        ball_pos = state.ball.position
        rewards = {}

        for agent in agents:
            car = state.cars[agent]
            car_pos = car.physics.position
            car_vel = car.physics.linear_velocity

            # direção inicial até a bola
            initial_dir = ball_pos - self.start_positions[agent]
            dist_initial = np.linalg.norm(initial_dir)
            if dist_initial > 1e-6:
                initial_dir /= dist_initial
            else:
                initial_dir = np.array([0.0, 0.0, 0.0])

            # distância atual até a bola
            dist_now = np.linalg.norm(ball_pos - car_pos)
            if dist_now > 1e-6:
                dir_to_ball = (ball_pos - car_pos) / dist_now

                # velocidade na direção da bola
                speed_reward = max(np.dot(car_vel, dir_to_ball) / common_values.CAR_MAX_SPEED, 0.0)

                # alinhamento apenas com a reta inicial
                alignment_reward = (np.dot(car.physics.forward, initial_dir) + 1.0) / 2.0

                # penalidade lateral MAIS FORTE
                lateral_vector = car_pos - (self.start_positions[agent] +
                                            initial_dir * np.dot(car_pos - self.start_positions[agent], initial_dir))
                lateral_distance = np.linalg.norm(lateral_vector)
                lateral_penalty = -2.0 * (1 / (1 + np.exp(-5 * (lateral_distance - 0.5))))
            else:
                speed_reward = 0.0
                alignment_reward = 0.5
                lateral_penalty = 0.0

            # progresso até a bola
            progress = max((self.start_dist[agent] - dist_now) / self.start_dist[agent], 0.0)

            # decaimento temporal (~100 ticks ≈ 1.6s)
            time_decay = max(0.1, 1 - self.tick_count / 100)

            # recompensa base
            reward = (progress + speed_reward + alignment_reward) * time_decay + lateral_penalty

            # ---- Recompensa por uso de boost ----
            if car.is_boosting:
                reward += 1.0  # ajustável

            # ---- Bônus por primeiro toque ----
            if car.ball_touches > 0 and not self.first_touch[agent]:
                self.first_touch[agent] = True
                reward += 2.0  # mantido pequeno para não distorcer demais
                self.kickoff_active = False  # termina kickoff após o(s) primeiro(s) toque(s)

            rewards[agent] = reward

        return rewards
