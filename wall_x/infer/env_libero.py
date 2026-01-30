import os
import json
import numpy as np
from typing import Dict, Any, Tuple, List
from libero.libero import benchmark
from wall_x.infer.env import BaseEnv, InferConfig
from wall_x.serving.policy.wall_x_policy import WallXPolicy

from wall_x.infer.base_dataclass import RobotStateActionData
from wall_x.infer.utils_libero import (
    get_libero_env,
    get_libero_dummy_action,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    TaskSuite,
    save_rollout_video,
)
from robosuite.wrappers import VisualizationWrapper


def _create_libero_env_standalone(
    task_id: int,
    task_suite_name: str,
    model_family: str = "wallx",
    resolution: int = 256,
    seed: int = 7,
) -> Any:
    """
    Standalone function to create a Libero environment, independent of LiberoRobotEnv instance.
    Used for creating environments in subprocess during multi-batch inference,
    avoiding serialization of large objects containing the model.

    Args:
        task_id: Task ID
        task_suite_name: Task suite name
        model_family: Model family
        resolution: Resolution
        seed: Random seed

    Returns:
        Environment instance
    """
    from libero.libero import benchmark

    # Get task suite and task
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)

    # Create environment
    env, _ = get_libero_env(
        task,
        model_family=model_family,
        resolution=resolution,
        seed=seed,
    )

    # Wrap environment
    env.env = VisualizationWrapper(env.env)
    env.env.set_visualization_setting(setting="grippers", visible=False)

    return env


class LiberoRobotEnv(BaseEnv):
    def __init__(
        self,
        config: InferConfig,
        task_suite_name: str = TaskSuite.LIBERO_SPATIAL,
        initial_states_path: str = "DEFAULT",
        rollout_dir: str = "./rollouts",
        model_family: str = "wallx",
        resolution: int = 256,
        seed: int = 7,
    ):

        super().__init__(config)
        self.logger.info(
            f"Initializing LiberoRobotEnv (Stateless), task suite: {task_suite_name}"
        )

        self.model = self._register_model()
        self.model_family = model_family
        self.resolution = resolution
        self.seed = seed

        self.logger.info("Importing Libero and related utils...")
        self.RobotStateActionData = RobotStateActionData

        self.rollout_dir = os.path.join(rollout_dir, task_suite_name)
        os.makedirs(self.rollout_dir, exist_ok=True)

        if save_rollout_video is not None:
            self.save_rollout_video = save_rollout_video
        else:
            self.save_rollout_video = None
            self.logger.warning("save_rollout_video not found, video saving disabled.")

        self.task_suite_name = task_suite_name
        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[self.task_suite_name]()
        self.num_tasks = self.task_suite.n_tasks

        self.initial_states_path = initial_states_path
        self.all_initial_states = None
        if self.initial_states_path != "DEFAULT":
            try:
                with open(self.initial_states_path, "r") as f:
                    self.all_initial_states = json.load(f)
                self.logger.info(f"Loaded custom initial states from {self.initial_states_path}")
            except Exception as e:
                self.logger.error(f"Failed to load initial states file: {e}")
                raise

    def _register_model(self) -> WallXPolicy:

        return WallXPolicy(model_path = self.config.model_path,
                            train_config = self.config.train_config,
                            action_tokenizer_path = self.config.action_tokenizer_path,
                            action_dim = self.config.action_dim,
                            agent_pos_dim= self.config.action_dim,
                            pred_horizon= self.config.pred_horizon,
                            camera_key= self.config.cam_names,
                            predict_mode=self.config.predict_mode)

    def get_instruction(self, task_desc: str) -> str:
        return task_desc

    def get_observation(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        if raw_obs is None:
            raise ValueError("Raw observation is None")

        data_obj = self.RobotStateActionData(config=self.config)

        pos = raw_obs["robot0_eef_pos"]
        rot = quat2axisangle(raw_obs["robot0_eef_quat"])
        grip = raw_obs["robot0_gripper_qpos"][0:1]

        data_obj.save_state_data_with_key(pos[None], "follow_right_ee_cartesian_pos")
        data_obj.save_state_data_with_key(rot[None], "follow_right_ee_rotation")
        data_obj.save_state_data_with_key(grip[None], "follow_right_gripper")
        data_obj.dof_mask = self._get_dof_mask()

        face_view = get_libero_image(raw_obs)
        right_wrist_view = get_libero_wrist_image(raw_obs)

        return {
            "robot_state_action_data": data_obj,
            "face_view": face_view,
            "right_wrist_view": right_wrist_view,
        }

    def apply_action(
        self, input_data: Dict[str, Any], env: Any = None, replay_images: list = None
    ) -> bool:
        if env is None:
            raise ValueError(
                "In Stateless mode, apply_action must be called with explicit 'env' parameter"
            )

        action_data = input_data["robot_state_action_data"]
        right_arm_traj = self._get_right_arm_action(action_data)
        while (
            right_arm_traj is not None
            and right_arm_traj.ndim > 2
            and right_arm_traj.shape[0] == 1
        ):
            right_arm_traj = right_arm_traj.squeeze(0)

        done = False
        t = 0

        try:
            for i in range(len(right_arm_traj)):
                if done:
                    break

                action_7d = right_arm_traj[i]
                obs, reward, done, info = env.step(action_7d)
                t += 1

                if obs is not None and replay_images is not None:
                    replay_images.append(get_libero_image(obs))

                input_data["_last_obs"] = obs

        except Exception as e:
            self.logger.error(f"Env step error: {e}")
            return False  # Error treated as failure

        return done, t

    def apply_action_batch(
        self,
        vec_env: Any,
        trajectories: List[np.ndarray],
        active_indices: List[int],
        status_list: List[Dict[str, Any]],
        model_outputs: List[Dict[str, Any]],
    ) -> None:
        """
        Execute action trajectories in parallel batch.

        Uses SubprocVectorEnv to execute actions in parallel for all active environments.
        Integrates vec_env.step(batch_actions, id=still_active) in this function.
        """
        if not trajectories:
            return

        max_traj_len = max(len(traj) for traj in trajectories)
        if max_traj_len == 0:
            return

        for step_idx in range(max_traj_len):
            # Check if there are still active environments
            still_active = [
                idx
                for idx in active_indices
                if (not status_list[idx]["done"])
                and status_list[idx]["count"] > 0
                and step_idx < len(trajectories[active_indices.index(idx)])
            ]
            if not still_active:
                break

            # Build batch actions (only includes actions for still_active environments)
            batch_actions = []
            for idx in still_active:
                traj_idx = active_indices.index(idx)
                action_7d = trajectories[traj_idx][step_idx]
                # Ensure action_7d is numpy array or list
                if isinstance(action_7d, np.ndarray):
                    batch_actions.append(action_7d)
                else:
                    batch_actions.append(np.array(action_7d))

            # Convert to numpy array with shape (batch_size, action_dim)
            batch_actions = np.array(batch_actions)

            # Execute step in parallel (only for still_active environments)
            obs_list, reward_list, done_list, info_list = vec_env.step(
                batch_actions, id=still_active
            )

            # Process returned results
            if obs_list.dtype == object:
                obs_list = list(obs_list)
            else:
                obs_list = [obs_list[i] for i in range(len(obs_list))]
            done_list = [bool(done_list[i]) for i in range(len(done_list))]

            # Update each environment's status
            for i, idx in enumerate(still_active):
                st = status_list[idx]
                obs = obs_list[i]
                done = done_list[i]

                if obs is not None:
                    st["current_obs"] = obs
                    if st["replay_images"] is not None:
                        st["replay_images"].append(get_libero_image(obs))

                st["count"] -= 1
                st["success"] = done
                st["done"] = done or st["count"] <= 0

                # Update model_output's _last_obs
                model_outputs[active_indices.index(idx)]["_last_obs"] = obs

    def get_task_info(self, task_id: int) -> Tuple[str, Any]:
        """
        Get task information (task description and initial states) without creating environment.

        Returns:
            Tuple[str, Any]: (task_desc, default_initial_states)
        """
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Invalid task ID: {task_id}")

        task = self.task_suite.get_task(task_id)
        task_desc = task.language
        default_initial_states = self.task_suite.get_task_init_states(task_id)

        return task_desc, default_initial_states

    def create_env_for_task(self, task_id: int) -> Tuple[Any, str, Any]:
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Invalid task ID: {task_id}")

        task = self.task_suite.get_task(task_id)
        default_initial_states = self.task_suite.get_task_init_states(task_id)

        env, task_desc = get_libero_env(
            task,
            model_family=self.model_family,
            resolution=self.resolution,
            seed=self.seed,
        )

        env.env = VisualizationWrapper(env.env)
        env.env.set_visualization_setting(setting="grippers", visible=False)

        return env, task_desc, default_initial_states

    def _get_initial_state_for_episode(
        self, task_desc: str, default_states: Any, episode_idx: int
    ) -> np.ndarray:
        if self.initial_states_path == "DEFAULT":
            if default_states is None:
                raise ValueError("Default states missing")
            return default_states[episode_idx]
        else:
            if self.all_initial_states is None:
                raise ValueError("Custom states not loaded")
            initial_states_task_key = task_desc.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"
            if not self.all_initial_states[initial_states_task_key][episode_key][
                "success"
            ]:
                raise ValueError(f"Expert demo failed for {episode_key}")
            return np.array(
                self.all_initial_states[initial_states_task_key][episode_key][
                    "initial_state"
                ]
            )

    def reset_env(
        self, env: Any, task_desc: str, default_states: Any, episode_idx: int
    ) -> Any:
        try:
            if episode_idx >= 0:
                state = self._get_initial_state_for_episode(
                    task_desc, default_states, episode_idx
                )
                obs = env.set_init_state(state)
                return obs
            else:
                return env.reset()
        except Exception as e:
            self.logger.error(f"Reset failed: {e}, falling back to default reset")
            return env.reset()

    def _get_dof_mask(self):
        dof_config = self.config.train_config["dof_config"]
        total_dof = sum(dof_config.values())
        dof_mask = np.ones((1, self.config.action_horizon, total_dof))
        mask_keys = [
            "follow_left_ee_cartesian_pos",
            "follow_left_ee_rotation",
            "follow_left_gripper",
            "head_actions",
            "height",
            "velocity_decomposed",
        ]
        start_idx = 0
        for key, dof_size in dof_config.items():
            if key in mask_keys:
                dof_mask[:, :, start_idx : start_idx + dof_size] = 0
            start_idx += dof_size
        return dof_mask

    def _get_right_arm_action(
        self, robot_state_action_data: RobotStateActionData
    ) -> np.ndarray:
        right_ee_cartesian_pos = robot_state_action_data.data[
            "action_right_ee_cartesian_pos"
        ]
        right_ee_rotation = robot_state_action_data.data["action_right_ee_rotation"]
        right_gripper = robot_state_action_data.data["action_right_gripper"]
        return np.concatenate(
            [right_ee_cartesian_pos, right_ee_rotation, right_gripper], axis=1
        )

    def _get_left_arm_action(
        self, robot_state_action_data: RobotStateActionData
    ) -> np.ndarray:
        left_ee_cartesian_pos = robot_state_action_data.data[
            "action_left_ee_cartesian_pos"
        ]
        left_ee_rotation = robot_state_action_data.data["action_left_ee_rotation"]
        left_gripper = robot_state_action_data.data["action_left_gripper"]
        return np.concatenate(
            [left_ee_cartesian_pos, left_ee_rotation, left_gripper], axis=1
        )

    def _save_rollout(
        self,
        replay_images: List[np.ndarray],
        success: bool,
        task_id: int,
        task_desc: str,
        episode_idx: int,
    ):
        if not self.save_rollout_video or not replay_images:
            return
        try:
            task_name_safe = task_desc.replace(" ", "_").replace(".", "")
            filename = f"{episode_idx}{'_SUCCESS' if success else '_FAILURE'}--_{task_name_safe}.mp4"
            self.save_rollout_video(
                self.rollout_dir,
                replay_images,
                filename,
                success=success,
                task_description=task_desc,
                log_file=None,
                model_family=self.model_family,
            )
            self.logger.info(f"Saved video: {filename}")
        except Exception as e:
            self.logger.error(f"Save video failed: {e}")

    def run_infer_flow_action(
        self,
        env: Any,
        task_id: int,
        task_desc: str,
        default_initial_states: Any,
        episode_idx: int,
        max_infer_times: int = 5,
        num_steps_wait: int = 10,
    ) -> bool:
        replay_images = []
        num_steps = 0
        done = False
        count = max_infer_times

        current_obs = self.reset_env(
            env, task_desc, default_initial_states, episode_idx
        )
        if current_obs is None:
            return False

        while num_steps < num_steps_wait:
            obs, reward, done, info = env.step(
                get_libero_dummy_action(self.model_family)
            )
            num_steps += 1
            if obs is not None:
                current_obs = obs

        while not done and count > 0:
            try:
                model_input = self.get_observation(current_obs)
                instruction = self.get_instruction(task_desc)
                model_input["prompt"] = instruction
                model_input["dataset_names"] = "libero_all"

                state = np.concatenate([
                    model_input['robot_state_action_data'].data["state_right_ee_cartesian_pos"],
                    model_input['robot_state_action_data'].data["state_right_ee_rotation"],
                    model_input['robot_state_action_data'].data["state_right_gripper"]
                ],axis=-1)

                model_input["state"] = state
                model_output = self.model.infer(model_input)

                model_output["robot_state_action_data"] = model_input["robot_state_action_data"]
                model_output["robot_state_action_data"].save_action_data(
                    model_output["predict_action"]
                )

                model_output["_last_obs"] = None

                done, delta_t = self.apply_action(
                    model_output, env=env, replay_images=replay_images
                )

                if model_output.get("_last_obs") is not None:
                    current_obs = model_output["_last_obs"]

                count -= delta_t

            except Exception as e:
                self.logger.error(f"Episode Error: {e}")
                import traceback

                traceback.print_exc()
                break

        success = done
        if count <= 0 and not done:
            self.logger.warning(
                f"Timeout: reached {max_infer_times} steps without success."
            )
            success = False

        self._save_rollout(replay_images, success, task_id, task_desc, episode_idx)
        return success

    def run_infer_flow_action_batch(
        self,
        vec_env: Any,
        task_ids: List[int] = None,
        task_descs: List[str] = None,
        default_initial_states_list: List[Any] = None,
        episode_indices: List[int] = None,
        max_infer_times: int = 5,
        num_steps_wait: int = 10,
    ) -> List[bool]:
        """
        Support batch inference: model inference in parallel (batch), environment execution in parallel (SubprocVectorEnv).

        Uses SubprocVectorEnv to run environments in subprocess during multi-batch inference, all environments execute actions in parallel.

        Returns a list of success flags for each sample.
        """
        if vec_env is None:
            raise ValueError("vec_env must be specified")
        batch_size = len(vec_env)
        if task_ids is not None:
            assert len(task_ids) == batch_size, "task_ids length must match envs"
        if episode_indices is not None:
            assert (
                len(episode_indices) == batch_size
            ), "episode_indices length must match envs"

        status_list = []
        for i in range(batch_size):
            status_list.append(
                {
                    "vec_env": vec_env,
                    "env_id": i,  # Index in vec_env
                    "task_desc": task_descs[i],
                    "replay_images": [],
                    "num_steps": 0,
                    "done": False,  # Whether episode has ended
                    "success": False,  # Whether successfully completed
                    "count": max_infer_times,
                    "current_obs": None,
                    "default_states": None,
                }
            )

        # Initialize/reset: Use SubprocVectorEnv to batch set initial states
        init_states_to_set = []
        for i in range(batch_size):
            task_desc = task_descs[i]
            default_states = default_initial_states_list[i]
            status_list[i]["default_states"] = default_states
            ep_i = episode_indices[i]
            init_state = self._get_initial_state_for_episode(
                task_desc, default_states, ep_i
            )
            init_states_to_set.append(init_state)

        # Batch set initial states
        try:
            obs_list = vec_env.set_init_state(init_states_to_set)
            if obs_list.dtype == object:
                obs_list = list(obs_list)
            else:
                obs_list = [obs_list[i] for i in range(len(obs_list))]

            for i, obs in enumerate(obs_list):
                if obs is None:
                    raise ValueError(
                        f"Reset environment returned None, task_id: {task_ids[i]}, episode_idx: {episode_indices[i]}"
                    )
                status_list[i]["current_obs"] = obs
                status_list[i]["done"] = False
                status_list[i]["success"] = False
                status_list[i]["count"] = max_infer_times
        except Exception as e:
            self.logger.error(f"Failed to batch set initial states: {e}")
            raise

        # Warmup steps (batch execution)
        dummy_action = get_libero_dummy_action(self.model_family)
        dummy_actions = np.array([dummy_action] * batch_size)
        for _ in range(num_steps_wait):
            obs_list, _, done_list, _ = vec_env.step(dummy_actions)
            # Update current_obs
            if obs_list.dtype == object:
                obs_list = list(obs_list)
            else:
                obs_list = [obs_list[i] for i in range(len(obs_list))]
            for i, obs in enumerate(obs_list):
                if obs is not None:
                    status_list[i]["current_obs"] = obs

        # Main loop: model parallel inference, environment parallel execution (SubprocVectorEnv)
        while any((not st["done"]) and st["count"] > 0 for st in status_list):
            active_indices = [
                idx
                for idx, st in enumerate(status_list)
                if (not st["done"]) and st["count"] > 0
            ]
            if not active_indices:
                break
            print(f"Batch infer loop, active indices: {active_indices}")

            observations = []
            instructions = []
            for idx in active_indices:
                st = status_list[idx]
                observations.append(self.get_observation(st["current_obs"]))
                instructions.append(self.get_instruction(st["task_desc"]))

            # Model batch inference
            model_outputs = self.model.infer_flow_action_batch(
                observations, instructions
            )
            # Extract action trajectories for all active environments
            trajectories = []
            for out in model_outputs:
                action_data = out["robot_state_action_data"]
                right_arm_traj = self._get_right_arm_action(action_data)
                while (
                    right_arm_traj is not None
                    and right_arm_traj.ndim > 2
                    and right_arm_traj.shape[0] == 1
                ):
                    right_arm_traj = right_arm_traj.squeeze(0)
                if right_arm_traj is None or len(right_arm_traj) == 0:
                    # If trajectory is empty, create an empty trajectory
                    right_arm_traj = np.array([]).reshape(0, 7)
                trajectories.append(right_arm_traj)

            # Use apply_action_batch to execute action trajectories in parallel
            try:
                self.apply_action_batch(
                    vec_env=vec_env,
                    trajectories=trajectories,
                    active_indices=active_indices,
                    status_list=status_list,
                    model_outputs=model_outputs,
                )
            except Exception as e:
                self.logger.error(f"Batch parallel action error: {e}")
                # Mark all active environments as failed
                for idx in active_indices:
                    status_list[idx]["done"] = True
                    status_list[idx]["success"] = False

        # Save replay and results
        success_list = []
        for i, st in enumerate(status_list):
            success = st.get("success", False)
            if st["count"] <= 0 and not st["success"]:
                self.logger.warning(
                    f"Batch timeout: reached {max_infer_times} steps without success (idx {i})."
                )
            st["replay_images"] = st.get("replay_images", [])
            tid_i = task_ids[i]
            epi_i = episode_indices[i]
            self._save_rollout(
                st["replay_images"],
                success,
                tid_i,
                st["task_desc"],
                epi_i,
            )
            success_list.append(success)

        return success_list

    def run_infer_ar_action(
        self,
        env: Any,
        task_id: int,
        task_desc: str,
        default_initial_states: Any,
        episode_idx: int,
        max_infer_times: int = 10,
        num_steps_wait: int = 10,
    ) -> bool:
        replay_images = []
        num_steps = 0
        done = False
        count = max_infer_times

        current_obs = self.reset_env(
            env, task_desc, default_initial_states, episode_idx
        )
        if current_obs is None:
            self.logger.error("Environment reset returned None.")
            return False

        while num_steps < num_steps_wait:
            obs, reward, done, info = env.step(
                get_libero_dummy_action(self.model_family)
            )
            num_steps += 1
            if obs is not None:
                current_obs = obs

        while not done and count > 0:
            try:
                model_input = self.get_observation(current_obs)
                instruction = self.get_instruction(task_desc)

                model_output = self.model.infer_ar_action(model_input, instruction)

                model_output["_last_obs"] = None

                done, delta_t = self.apply_action(
                    model_output, env=env, replay_images=replay_images
                )

                if model_output.get("_last_obs") is not None:
                    current_obs = model_output["_last_obs"]
                else:
                    if not done:
                        self.logger.warning(
                            "Did not receive new observation after apply_action, but episode is not done."
                        )

                count -= delta_t

            except Exception as e:
                self.logger.error(
                    f"AR Episode Run Error (Task {task_id}, Ep {episode_idx}): {e}"
                )
                import traceback

                traceback.print_exc()
                break

        success = done
        if count <= 0 and not done:
            self.logger.warning(
                f"Timeout: AR policy reached {max_infer_times} steps without success."
            )
            success = False

        self._save_rollout(replay_images, success, task_id, task_desc, episode_idx)

        return success
