import socket
import struct
import json
import cv2
import numpy as np
import time
import select
import errno
import os
import pickle

import threading

from wall_x._vendor.harrix.serving._wallx_infer.logger import InferLogger

_DEFAULT_ROBOT_CONFIG = {
    0: {"host": "127.0.0.1", "action_port": 57749, "keyboard_port": 58849},
    1: {"host": "127.0.0.1", "action_port": 57750, "keyboard_port": 58850},
    2: {"host": "127.0.0.1", "action_port": 57751, "keyboard_port": 58851},
    3: {"host": "127.0.0.1", "action_port": 57761, "keyboard_port": 58861},
}


class RobotRegistry:
    def __init__(self):
        self.registry = _DEFAULT_ROBOT_CONFIG
        self.logger = InferLogger.get_controller_logger("RobotRegistry")
        self.logger.debug(f"init robot registry, registered {len(self.registry)} robots")

    def register_robot(self, robot_id, host, action_port, keyboard_port):
        if robot_id in self.registry:
            self.logger.warning(f"robot ID {robot_id} already registered")
            return False
        else:
            self.registry[robot_id] = {
                "host": host,
                "action_port": action_port,
                "keyboard_port": keyboard_port,
            }
            self.logger.info(
                f"register robot ID={robot_id}: host={host}, action_port={action_port}, keyboard_port={keyboard_port}"
            )
            return True

    def get_robot_info(self, robot_id):
        if robot_id in self.registry:
            info = self.registry[robot_id]
            self.logger.debug(f"get robot ID={robot_id} info: {info}")
            return info
        else:
            self.logger.error(f"robot ID={robot_id} not found")
            return None

    def exist(self, robot_id):
        return robot_id in self.registry


class RobotCommunication:
    def __init__(self, robot_id, host=None, action_port=None, keyboard_port=None):
        self.logger = InferLogger.get_controller_logger("RobotCommunication")
        self.logger.debug(f"init robot comms: robot_id={robot_id}")

        self.robot_register = RobotRegistry()
        if self.robot_register.exist(robot_id):
            self.robot_info = self.robot_register.get_robot_info(robot_id)
            self.logger.debug(f"using registered robot config: {robot_id}")
        else:
            self.robot_register.register_robot(
                robot_id, host, action_port, keyboard_port
            )
            self.robot_info = self.robot_register.get_robot_info(robot_id)
            self.logger.debug(f"register new robot config: {robot_id}")

        self.action_sock = None
        self.action_conn = None

        self.keyboard_sock = None
        self.keyboard_conn = None

        self.client_socks = []

    def connect(self):
        self.logger.info("establishing socket connection...")

        self.action_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.action_sock.setblocking(True)
        self.action_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        host = self.robot_info["host"]
        port = self.robot_info["action_port"]
        self.action_sock.bind((host, port))
        self.action_sock.listen(1)
        self.logger.info(f"listening on action port: {host}:{self.action_sock.getsockname()[1]}")

        action_thread = threading.Thread(target=self.handle_action_client)

        self.logger.debug("starting connection handler thread...")
        action_thread.start()

        action_thread.join()
        self.logger.info("socket connection established")

    def handle_action_client(self):
        self.logger.debug("waiting for client connection...")
        self.action_conn, addr = self.action_sock.accept()
        self.logger.info(f"accepted connection from {addr}")

    def recv_image(self, index):
        self.logger.debug(f"receiving image index={index}...")
        image_size = struct.unpack("<L", self.action_conn.recv(4))[0]
        self.logger.debug(f"image size: {image_size} bytes")

        image = self.recvall(self.action_conn, image_size)
        nparr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.logger.debug(f"image decoded: shape={image.shape}")
        return image

    def recv_action_data(self):
        self.logger.debug("receiving action data...")
        data_size = struct.unpack("<L", self.action_conn.recv(4))[0]
        self.logger.debug(f"action data size: {data_size} bytes")

        data = self.recvall(self.action_conn, data_size)
        action_data = json.loads(data.decode("utf8"))

        self.logger.debug(f"action data received: {len(action_data)} fields")
        return action_data

    def accept_connections(self):
        try:
            client_sock, addr = self.keyboard_sock.accept()
            client_sock.setblocking(0)  # Set non-blocking mode for the client socket
            self.client_socks.append(client_sock)
        except BlockingIOError:
            # The socket is in non-blocking mode and there are no pending connections
            pass

    def recv_keyboard_input(self):
        for client_sock in list(self.client_socks):  # Iterate over a copy of the list
            if client_sock.fileno() == -1:
                # Socket has been closed, remove it from the list
                self.client_socks.remove(client_sock)
                continue
            read_sockets, _, _ = select.select([client_sock], [], [], 0)
            if client_sock in read_sockets:
                try:
                    data_size_bytes = client_sock.recv(4)
                    if not data_size_bytes:
                        self.logger.debug("client closed connection")
                        client_sock.close()
                        self.client_socks.remove(client_sock)
                        continue
                    data_size = struct.unpack("<L", data_size_bytes)[0]
                    data = RobotCommunication.recvall(client_sock, data_size)
                    if not data:
                        self.logger.debug("client closed connection")
                        client_sock.close()
                        self.client_socks.remove(client_sock)
                        continue
                    json_data = json.loads(data.decode("utf8"))
                    self.logger.debug(f"keyboard input: {json_data}")
                    return json_data
                except socket.error as e:
                    if e.errno == errno.EAGAIN or e.errno == errno.EWOULDBLOCK:
                        # Resource temporarily unavailable, ignore the error
                        pass
                    else:
                        self.logger.error(f"socket error: {e}")
                        client_sock.close()
                        self.client_socks.remove(client_sock)
                        continue
        time.sleep(0.1)
        return None

    def send_dict(self, dict_data):
        self.logger.debug(f"sending dict: {len(dict_data)} fields")
        data_str = json.dumps(dict_data)
        data_bytes = data_str.encode("utf-8")
        self.action_conn.sendall(struct.pack("<L", len(data_bytes)))
        self.action_conn.sendall(data_bytes)
        self.logger.debug(f"data sent: {len(data_bytes)} bytes")

    def close(self):
        self.logger.info("closing socket connection...")
        self.action_sock.close()
        for sock in self.client_socks:
            sock.close()
        self.logger.info(f"closed {len(self.client_socks)} client connections")
        # self.keyboard_sock.close()

    @staticmethod
    def recvall(sock, count):
        buf = b""
        while count:
            newbuf = sock.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf


class DummyRobotCommunication:
    def __init__(self):
        pass

    def send_dict(self, dict_data):
        pass


class DummyRobotController:
    def __init__(
        self,
        robot_id: int,
        host: str = None,
        port: str = None,
        views_path: str | None = None,
        state_path: str | None = None,
    ):
        self.robot_id = robot_id
        self.host = host
        self.port = port
        self.views_path = views_path or os.environ.get("WALL_X_DUMMY_VIEWS_PATH")
        self.state_path = state_path or os.environ.get("WALL_X_DUMMY_STATE_PATH")
        self.robot_comm = DummyRobotCommunication()
        self.logger = InferLogger.get_controller_logger("DummyRobotController")
        self.logger.info(
            f"init virtual robot controller (debug): robot_id={robot_id}"
        )

    def connect(self):
        self.logger.debug("virtual controller connect (no-op)")
        pass

    def close(self):
        self.logger.debug("virtual controller close (no-op)")
        pass

    def recv_image(
        self, cam_names: list = ["camera_left", "camera_front", "camera_right"]
    ) -> dict:
        if not self.views_path:
            raise RuntimeError(
                "DummyRobotController requires views_path or "
                "WALL_X_DUMMY_VIEWS_PATH."
            )
        self.logger.debug(f"load views from pickle: {cam_names}")
        with open(self.views_path, "rb") as f:
            all_views = pickle.load(f)
        self.logger.debug(f"views loaded: {len(all_views)} views")
        return all_views

    def recv_action(self):
        if not self.state_path:
            raise RuntimeError(
                "DummyRobotController requires state_path or "
                "WALL_X_DUMMY_STATE_PATH."
            )
        self.logger.debug("load actions from pickle")
        with open(self.state_path, "rb") as f:
            all_actions = pickle.load(f)
        self.logger.debug(f"actions loaded: {len(all_actions)} fields")
        return all_actions


class RobotController:
    def __init__(
        self,
        robot_id: int,
        host: str = None,
        port: str = None,
        max_time_step: int = 10000,
    ):
        self.logger = InferLogger.get_controller_logger("RobotController")
        self.logger.info(
            f"init robot controller: robot_id={robot_id}, max_time_step={max_time_step}"
        )

        self.robot_comm = RobotCommunication(robot_id, host, port)
        self.max_time_step = max_time_step
        self.global_step = 0

    def connect(self):
        self.logger.info("connecting robot controller...")
        self.robot_comm.connect()
        self.logger.info("robot controller connected")

    def close(self):
        self.logger.info("closing robot controller...")
        self.robot_comm.close()
        self.logger.info("robot controller closed")

    def prediction(self, views, actions) -> dict:
        raise NotImplementedError

    def recv_image(
        self, cam_names: list = ["camera_left", "camera_front", "camera_right"]
    ) -> dict:
        self.logger.debug(f"receive images: {cam_names}")
        views = {}
        for i, name in enumerate(cam_names):
            image = np.array(self.robot_comm.recv_image(i))
            views[name] = image[None, :]
            self.logger.debug(f"received {name}: shape={image.shape}")
        self.logger.debug(f"all images received: {len(views)} views")
        return views

    def recv_action(self):
        self.logger.debug("receiving action data...")
        action_data = self.robot_comm.recv_action_data()
        return action_data

    def recv_keyboard_input(self):
        self.robot_comm.accept_connections()
        json_data = self.robot_comm.recv_keyboard_input()
        if json_data is not None:
            self.logger.debug(
                f"keyboard: motionStatus={json_data.get('motionStatus')}, armMode={json_data.get('armMode')}"
            )
            return json_data["motionStatus"], json_data["armMode"]

        return None, None

    def reset(self):
        self.logger.info(f"reset controller: global_step {self.global_step} -> 0")
        self.global_step = 0

    def record_start(self):
        self.logger.info("start recording...")
        record_signal = {"cmd": "RECORD_START"}
        self.robot_comm.send_dict(record_signal)

    def record_continue(self):
        self.logger.debug("continue recording...")
        record_signal = {"cmd": "RECORD_CONTINUE"}
        self.robot_comm.send_dict(record_signal)

    def record_start_process(self):
        if self.global_step == 0:
            self.logger.info("waiting for START signal...")
            action = None
            while action != "START":
                action, _ = self.recv_keyboard_input()
            self.logger.info("received START signal")
            self.record_start()

        self.record_continue()

    def set_zero(self):
        self.logger.info("setting zero pose...")
        record_signal = {"cmd": "INIT_ZERO", "gripper": [0.0, 0.0]}
        self.robot_comm.send_dict(record_signal)
        self.reset()
        self.logger.info("zero pose set")

    def record_stop(self):
        self.logger.info("stop recording...")
        record_signal = {"cmd": "RECORD_STOP"}
        self.robot_comm.send_dict(record_signal)
        time.sleep(0.01)

    def recover_from_failure(self):
        self.logger.warning("recovering from failure...")
        record_signal = {"cmd": "TO_MASTER_SLAVE"}
        self.robot_comm.send_dict(record_signal)
        time.sleep(0.01)

        self.logger.info("waiting for START signal...")
        action = None
        while action != "START":
            action, _ = self.recv_keyboard_input()

        self.logger.info("recovery done, resume recording")
        self.record_start()

    def reset_to_zero(self):
        self.logger.info("reset to zero pose...")
        self.record_stop()
        self.set_zero()

    def to_slave(self):
        self.logger.info("switching to slave mode...")
        record_signal = {"cmd": "TO_SLAVE"}
        self.robot_comm.send_dict(record_signal)
        time.sleep(0.01)

    def run(
        self,
        record_mode=False,
        cam_names: list = ["camera_left", "camera_front", "camera_right"],
    ):

        while self.global_step <= self.max_time_step:
            if record_mode:
                self.record_start_process()

            self.global_step += 1

            action = self.recv_action()
            view = self.recv_image(cam_names)

            pred = self.prediction(view, action)
            self.robot_comm.send_dict(pred)

            if record_mode:
                action, arm_mode = self.recv_keyboard_input()

                if action is not None and arm_mode is not None:
                    self.logger.info("action: %s, arm_mode: %s", action, arm_mode)
                    if action == "STOP" and arm_mode == "ARM_TEST_MODE_MS":
                        self.record_stop()
                        self.recover_from_failure()
                        action, arm_mode = None, None
                        while action != "STOP" or arm_mode != "ARM_TEST_MODE_S":
                            action, arm_mode = self.recv_keyboard_input()
                        self.record_stop()
                        self.to_slave()
                        self.reset()
                    elif arm_mode == "ARM_TEST_MODE_S" and action == "INIT":
                        self.reset_to_zero()
