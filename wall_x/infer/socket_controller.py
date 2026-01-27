import socket
import struct
import json
import cv2
import numpy as np
import time
import select
import errno
import pickle

import threading

from wall_x.infer.logger import InferLogger

_DEFAULT_ROBOT_CONFIG = {
    0: {"host": "192.168.100.53", "action_port": 57749, "keyboard_port": 58849},
    1: {"host": "192.168.100.53", "action_port": 57750, "keyboard_port": 58850},
    2: {"host": "192.168.100.53", "action_port": 57751, "keyboard_port": 58851},
    3: {"host": "192.168.100.53", "action_port": 57761, "keyboard_port": 58861},
}


class RobotRegistry:
    def __init__(self):
        self.registry = _DEFAULT_ROBOT_CONFIG
        self.logger = InferLogger.get_controller_logger("RobotRegistry")
        self.logger.debug(f"初始化机器人注册表,已注册 {len(self.registry)} 个机器人")

    def register_robot(self, robot_id, host, action_port, keyboard_port):
        if robot_id in self.registry:
            self.logger.warning(f"机器人ID {robot_id} 已被注册")
            return False
        else:
            self.registry[robot_id] = {
                "host": host,
                "action_port": action_port,
                "keyboard_port": keyboard_port,
            }
            self.logger.info(
                f"注册机器人 ID={robot_id}: host={host}, action_port={action_port}, keyboard_port={keyboard_port}"
            )
            return True

    def get_robot_info(self, robot_id):
        if robot_id in self.registry:
            info = self.registry[robot_id]
            self.logger.debug(f"获取机器人 ID={robot_id} 信息: {info}")
            return info
        else:
            self.logger.error(f"机器人 ID={robot_id} 未找到")
            return None

    def exist(self, robot_id):
        return robot_id in self.registry


class RobotCommunication:
    def __init__(self, robot_id, host=None, action_port=None, keyboard_port=None):
        self.logger = InferLogger.get_controller_logger("RobotCommunication")
        self.logger.debug(f"初始化机器人通信: robot_id={robot_id}")

        self.robot_register = RobotRegistry()
        if self.robot_register.exist(robot_id):
            self.robot_info = self.robot_register.get_robot_info(robot_id)
            self.logger.debug(f"使用已注册的机器人配置: {robot_id}")
        else:
            self.robot_register.register_robot(
                robot_id, host, action_port, keyboard_port
            )
            self.robot_info = self.robot_register.get_robot_info(robot_id)
            self.logger.debug(f"注册新机器人配置: {robot_id}")

        self.action_sock = None
        self.action_conn = None

        self.keyboard_sock = None
        self.keyboard_conn = None

        self.client_socks = []

    def connect(self):
        self.logger.info("开始建立Socket连接...")

        self.action_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.action_sock.setblocking(True)
        self.action_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        host = self.robot_info["host"]
        port = self.robot_info["action_port"]
        self.action_sock.bind((host, port))
        self.action_sock.listen(1)
        self.logger.info(f"监听动作端口: {host}:{self.action_sock.getsockname()[1]}")

        # self.keyboard_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.keyboard_sock.setblocking(False) # 非阻塞
        # self.keyboard_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # self.keyboard_sock.bind((self.robot_info['host'], self.robot_info['keyboard_port']))
        # self.keyboard_sock.listen(1)
        # self.logger.info(f"监听键盘端口: {self.keyboard_sock.getsockname()[1]}")

        action_thread = threading.Thread(target=self.handle_action_client)
        # keyboard_thread = threading.Thread(target=self.handle_keyboard_client)

        self.logger.debug("启动连接处理线程...")
        action_thread.start()
        # keyboard_thread.start()

        action_thread.join()
        # keyboard_thread.join()
        self.logger.info("Socket连接已建立")

    def handle_action_client(self):
        self.logger.debug("等待客户端连接...")
        self.action_conn, addr = self.action_sock.accept()
        self.logger.info(f"接受来自 {addr} 的连接")

    # def handle_keyboard_client(self):
    #     try:
    #         self.keyboard_conn, addr = self.keyboard_sock.accept()
    #         self.keyboard_conn.setblocking(False)  # 设置为非阻塞模式
    #         print(f"Connection accepted from {addr}")
    #         # 处理连接
    #     except socket.error as e:
    #         if e.errno == errno.EAGAIN or e.errno == errno.EWOULDBLOCK:
    #             # 资源暂时不可用,忽略错误
    #             pass
    #         else:
    #             print(f"Socket error: {e}")

    def recv_image(self, index):
        self.logger.debug(f"接收图像 index={index}...")
        image_size = struct.unpack("<L", self.action_conn.recv(4))[0]
        self.logger.debug(f"图像大小: {image_size} bytes")

        image = self.recvall(self.action_conn, image_size)
        nparr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.logger.debug(f"图像解码完成: shape={image.shape}")
        return image

    def recv_action_data(self):
        self.logger.debug("接收动作数据...")
        data_size = struct.unpack("<L", self.action_conn.recv(4))[0]
        self.logger.debug(f"动作数据大小: {data_size} bytes")

        data = self.recvall(self.action_conn, data_size)
        action_data = json.loads(data.decode("utf8"))

        self.logger.debug(f"动作数据接收完成: {len(action_data)} 个字段")
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
                        self.logger.debug("客户端关闭连接")
                        client_sock.close()
                        self.client_socks.remove(client_sock)
                        continue
                    data_size = struct.unpack("<L", data_size_bytes)[0]
                    data = RobotCommunication.recvall(client_sock, data_size)
                    if not data:
                        self.logger.debug("客户端关闭连接")
                        client_sock.close()
                        self.client_socks.remove(client_sock)
                        continue
                    json_data = json.loads(data.decode("utf8"))
                    self.logger.debug(f"接收到键盘输入: {json_data}")
                    return json_data
                except socket.error as e:
                    if e.errno == errno.EAGAIN or e.errno == errno.EWOULDBLOCK:
                        # Resource temporarily unavailable, ignore the error
                        pass
                    else:
                        self.logger.error(f"Socket错误: {e}")
                        client_sock.close()
                        self.client_socks.remove(client_sock)
                        continue
        time.sleep(0.1)
        return None

    def send_dict(self, dict_data):
        self.logger.debug(f"发送字典数据: {len(dict_data)} 个字段")
        data_str = json.dumps(dict_data)
        data_bytes = data_str.encode("utf-8")
        self.action_conn.sendall(struct.pack("<L", len(data_bytes)))
        self.action_conn.sendall(data_bytes)
        self.logger.debug(f"数据发送完成: {len(data_bytes)} bytes")

    def close(self):
        self.logger.info("关闭Socket连接...")
        self.action_sock.close()
        for sock in self.client_socks:
            sock.close()
        self.logger.info(f"已关闭 {len(self.client_socks)} 个客户端连接")
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
    def __init__(self, robot_id: int, host: str = None, port: str = None):
        self.robot_id = robot_id
        self.host = host
        self.port = port
        self.robot_comm = DummyRobotCommunication()
        self.logger = InferLogger.get_controller_logger("DummyRobotController")
        self.logger.info(
            f"初始化虚拟机器人控制器（用于debug测试）: robot_id={robot_id}"
        )

    def connect(self):
        self.logger.debug("虚拟控制器连接(无操作)")
        pass

    def close(self):
        self.logger.debug("虚拟控制器关闭(无操作)")
        pass

    def recv_image(
        self, cam_names: list = ["camera_left", "camera_front", "camera_right"]
    ) -> dict:
        # 从pickle文件加载视图数据
        self.logger.debug(f"从pickle文件加载视图数据: {cam_names}")
        with open(
            "/x2robot_v2/share/ryan/infer_example_data/desktop/views.pkl", "rb"
        ) as f:
            all_views = pickle.load(f)
        self.logger.debug(f"视图数据加载完成: {len(all_views)} 个视图")
        return all_views

    def recv_action(self):
        self.logger.debug("从pickle文件加载动作数据")
        with open(
            "/x2robot_v2/share/ryan/infer_example_data/desktop/state.pkl", "rb"
        ) as f:
            all_actions = pickle.load(f)
        self.logger.debug(f"动作数据加载完成: {len(all_actions)} 个字段")
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
            f"初始化机器人控制器: robot_id={robot_id}, max_time_step={max_time_step}"
        )

        self.robot_comm = RobotCommunication(robot_id, host, port)
        self.max_time_step = max_time_step
        self.global_step = 0

    def connect(self):
        self.logger.info("连接机器人控制器...")
        self.robot_comm.connect()
        self.logger.info("机器人控制器连接成功")

    def close(self):
        self.logger.info("关闭机器人控制器...")
        self.robot_comm.close()
        self.logger.info("机器人控制器已关闭")

    def prediction(self, views, actions) -> dict:
        raise NotImplementedError

    def recv_image(
        self, cam_names: list = ["camera_left", "camera_front", "camera_right"]
    ) -> dict:
        self.logger.debug(f"接收图像: {cam_names}")
        views = {}
        for i, name in enumerate(cam_names):
            image = np.array(self.robot_comm.recv_image(i))
            views[name] = image[None, :]
            self.logger.debug(f"接收 {name}: shape={image.shape}")
        self.logger.debug(f"所有图像接收完成: {len(views)} 个视图")
        return views

    def recv_action(self):
        self.logger.debug("接收动作数据...")
        action_data = self.robot_comm.recv_action_data()
        return action_data

    def recv_keyboard_input(self):
        self.robot_comm.accept_connections()
        json_data = self.robot_comm.recv_keyboard_input()
        if json_data is not None:
            self.logger.debug(
                f"键盘输入: motionStatus={json_data.get('motionStatus')}, armMode={json_data.get('armMode')}"
            )
            return json_data["motionStatus"], json_data["armMode"]

        return None, None

    def reset(self):
        self.logger.info(f"重置控制器: global_step {self.global_step} -> 0")
        self.global_step = 0

    def record_start(self):
        self.logger.info("开始录制...")
        record_signal = {"cmd": "RECORD_START"}
        self.robot_comm.send_dict(record_signal)

    def record_continue(self):
        self.logger.debug("继续录制...")
        record_signal = {"cmd": "RECORD_CONTINUE"}
        self.robot_comm.send_dict(record_signal)

    def record_start_process(self):
        if self.global_step == 0:
            self.logger.info("等待START信号...")
            action = None
            while action != "START":
                action, _ = self.recv_keyboard_input()
            self.logger.info("收到START信号")
            self.record_start()

        self.record_continue()

    def set_zero(self):
        self.logger.info("设置零位...")
        record_signal = {"cmd": "INIT_ZERO", "gripper": [0.0, 0.0]}
        self.robot_comm.send_dict(record_signal)
        self.reset()
        self.logger.info("零位设置完成")

    def record_stop(self):
        self.logger.info("停止录制...")
        record_signal = {"cmd": "RECORD_STOP"}
        self.robot_comm.send_dict(record_signal)
        time.sleep(0.01)

    def recover_from_failure(self):
        self.logger.warning("从失败中恢复...")
        record_signal = {"cmd": "TO_MASTER_SLAVE"}
        self.robot_comm.send_dict(record_signal)
        time.sleep(0.01)

        self.logger.info("等待START信号...")
        action = None
        while action != "START":
            action, _ = self.recv_keyboard_input()

        self.logger.info("恢复完成,重新开始录制")
        self.record_start()

    def reset_to_zero(self):
        self.logger.info("重置到零位...")
        self.record_stop()
        self.set_zero()

    def to_slave(self):
        self.logger.info("切换到从模式...")
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
                    print(f"action: {action}, arm_mode: {arm_mode}")
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
