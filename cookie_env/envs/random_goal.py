import numpy as np
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv


class GoalEnv(MiniGridEnv):
    """
    Entorno de navegación para MultiGoal.
    ------
    - Grilla cuadrada size x size, sin objetos Goal visibles.
    - El goal es un double one-hot float32 en el espacio de observación:
        onehot(row_idx, stoch_rows) ⊕ onehot(class_val, stoch_classes)
      Se samplea uniformemente al inicio de cada episodio y se mantiene fijo.
    - Reward: siempre -1 por step. El reward 0 (goal alcanzado) lo asigna
      driver.py al obtener el valor de z..
    - Terminación: solo por max_steps. La terminación por goal alcanzado
      la gestiona run_train.py truncando el buffer del episodio. 

    Parámetros
    ----------
    size          : tamaño de la grilla (incluye paredes)
    stoch_rows    : número de filas del z estocástico de DreamerV3
    stoch_classes : número de clases por fila del z estocástico
    agent_start_pos : posición inicial del agente (col, row). None = aleatorio
    agent_start_dir : dirección inicial (0=derecha, 1=abajo, 2=izquierda, 3=arriba)
    max_steps     : pasos máximos por episodio
    onehot        : si True, imagen en formato one-hot (H,W,5) uint8
    render_mode   : 'human' | 'rgb_array' | None
    """

    def __init__(
        self,
        size: int = 9,
        stoch_rows: int = 32,
        stoch_classes: int = 16,
        agent_start_pos=(1, 1),
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        onehot: bool = False,
        fixed_row: bool = False,
        render_mode: str | None = None,
        **kwargs,
    ):
        self.size             = size
        self.stoch_rows       = stoch_rows
        self.stoch_classes    = stoch_classes
        self.agent_start_pos  = agent_start_pos
        self.agent_start_dir  = agent_start_dir
        self.onehot           = onehot
        self.fixed_row        = fixed_row

        # goal y se inicializan en reset()
        if self.fixed_row:
            self._goal_dim = stoch_classes
        else:
            # Doble one-hot
            self._goal_dim = stoch_rows + stoch_classes

        # 8 acciones: 0-6 (MiniGrid) + 7 (nada)
        self.action_space = spaces.Discrete(8)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size * size

        super().__init__(
            mission_space=mission_space,
            height=size,
            width=size,
            see_through_walls=True,
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs,
        )

        # Ampliar observation_space con goal
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            'goal': spaces.Box(
                low=0.0, high=1.0,
                shape=(self._goal_dim,),
                dtype=np.float32,
            ),
        })
        if self.onehot:
            self._init_onehot_obs()

    # ── Observation space helpers ─────────────────────────────────────────
    IDX_TO_ONEHOT = {1: 0, 2: 1, 10: 2, 11: 3, 12: 4}
    def _init_onehot_obs(self):
        img_shape = self.observation_space['image'].shape
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            'image': spaces.Box(
                low=0, high=1,
                shape=(img_shape[0], img_shape[1], 5),
                dtype='uint8',
            ),
        })

    def _get_onehot_obs(self, obs):
        obj_types = obs['image'][:, :, 0].copy()
        obj_types[self.agent_view_size // 2, self.agent_view_size - 1] = 10
        onehot = np.zeros(
            (obs['image'].shape[0], obs['image'].shape[1], 5), dtype='uint8')
        for i in range(obj_types.shape[0]):
            for j in range(obj_types.shape[1]):
                idx = self.IDX_TO_ONEHOT.get(int(obj_types[i, j]))
                if idx is not None:
                    onehot[i, j, idx] = 1
        obs['image'] = onehot
        return obs

    # ── Goal helpers ──────────────────────────────────────────────────────

    def _sample_goal(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Samplea un goal double one-hot completamente aleatorio:
            row_idx   ~ U(0, stoch_rows)
            class_val ~ U(0, stoch_classes)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        class_val = int(rng.integers(self.stoch_classes))
        cls_oh = np.zeros(self.stoch_classes, dtype=np.float32)
        cls_oh[class_val] = 1.0
        if self.fixed_row:
            return cls_oh
        else:
            # Si es al azar, incluimos el one-hot de la fila seleccionada
            row_idx = int(rng.integers(self.stoch_rows))
            row_oh = np.zeros(self.stoch_rows, dtype=np.float32)
            row_oh[row_idx] = 1.0
            return np.concatenate([row_oh, cls_oh])

    @staticmethod
    def _gen_mission():
        return "reach the latent goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        # Sin objetos Goal visibles — el objetivo está en el espacio latente

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = self._gen_mission()

    def gen_obs(self):
        obs = super().gen_obs()
        if self.onehot:
            obs = self._get_onehot_obs(obs)
        obs['goal'] = self._goal.copy()
        return obs

    def reset(self, *, seed=None, options=None):
        # Samplear nuevo goal al inicio del episodio
        rng = np.random.default_rng(seed)
        self._goal     = self._sample_goal(rng)
        self._achieved = np.zeros(self.stoch_rows, dtype=np.int32)
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        if action == 7:
            self.step_count += 1
            obs = self.gen_obs()
            reward = -1.0
            terminated = False
            truncated = self.step_count >= self.max_steps
            return obs, reward, terminated, truncated, {}
        
        obs, reward, terminated, truncated, info = super().step(action)
        reward = -1.0
        terminated = False
        return obs, reward, terminated, truncated, info


if __name__ == '__main__':
    # python -m cookie_env.envs.random_goal
    from minigrid.manual_control import ManualControl
    import pygame
    
    env = GoalEnv(
        size=9, 
        stoch_rows=32, 
        stoch_classes=16,
        render_mode='human', 
        onehot=False,
        fixed_row=True
    )

    class HERManualControl(ManualControl):
        def __init__(self, env, seed=None):
            super().__init__(env, seed=seed)

        def key_handler(self, event):
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.step(7)
                    return
                
            super().key_handler(event)

    manual = HERManualControl(
        env, 
        seed=42
    )
    manual.start()