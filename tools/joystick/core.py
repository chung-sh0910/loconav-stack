import math
import time
from typing import Tuple

from ps4_data_parser import PS4DataParser


class PS4Controller(object):
    controller = None
    axis_data = {}
    button_data = {}
    hat_data = {}

    def __init__(self):
        vendor = 0x054C
        product = 0x0ba0
        self.data_parser: PS4DataParser = PS4DataParser(vendor, product)
        self.prev_connection_stable = True
        self.run_emergency_brake = False
        self.pub_go_waypoint = False
        self.pub_stop_waypoint = False
        self.battery_refresh_time = 0.
        self.min_battery_percenatage = 15
        self.current_battery_percenatage = 0.
        self.valid_action_timestamp = time.time()

        self.L_Xaxis = 0.
        self.L_Yaxis = 0.
        self.L_trigger = 0.

        self.R_Xaxis = 0.
        self.R_Yaxis = 0.
        self.R_trigger = 0.

        self.X_button = False
        self.O_button = False
        self.A_button = False
        self.H_button = False
        # Up, Down
        self.U_button = False
        self.D_button = False

    def check_emergency_brake(self) -> bool:
        if self.O_button:
            if self.run_emergency_brake:
                print("Emergency release!!")
            self.run_emergency_brake = False

        if self.X_button:
            if not self.run_emergency_brake:
                print("Emergency brake!!")
            self.run_emergency_brake = True

    def check_emergency_condition(self) -> bool:
        need_to_stop = self.run_emergency_brake or not self.prev_connection_stable
        need_to_shutdown = not self.has_enough_battery()
        if need_to_shutdown:
            print("Ps4 controller does not have enough battery")

        return need_to_shutdown, need_to_stop

    def check_pub_waypoint_command(self) -> bool:
        if self.U_button:
            self.pub_go_waypoint = True
        if self.D_button:
            self.pub_stop_waypoint = True
        return self.pub_go_waypoint, self.pub_stop_waypoint

    def get_robot_action(self) -> Tuple[bool, float, float, float]:
        return self._to_robot_action(self.R_trigger, self.L_trigger,
                                     self.L_Xaxis, self.R_Xaxis)

    def _to_robot_action(self, R_trigger: float, L_trigger: float,
                         L_Xaxis: float,
                         R_Xaxis: float) -> Tuple[bool, float, float, float]:
        # change joystick status to normalized velocity command (vx,w) -1.0~1.0
        ## key matching (dualshock4)
        # Forward: R_trigger # Backward: L_trigger # Rotate: R_Xaxis
        linear_action_x = (self.normalize_trigger_value(R_trigger) -
                           self.normalize_trigger_value(L_trigger))
        linear_action_y = self.normalize_axis_value(-L_Xaxis, margin=0.05)
        angular_action_w = self.normalize_axis_value(-R_Xaxis, margin=0.05)
        is_valid = self.check_actions(linear_action_x, linear_action_y,
                                      angular_action_w)
        return is_valid, linear_action_x, linear_action_y, angular_action_w

    def check_actions(self,
                      linear_action_x: float,
                      linear_action_y: float,
                      angular_action_w: float,
                      time_margin: float = 0.2) -> bool:
        actions = [linear_action_x, linear_action_y, angular_action_w]
        command_on = any(actions)
        if command_on:
            self.valid_action_timestamp = time.time()
            is_valid = True
            return is_valid
        time_diff = time.time() - self.valid_action_timestamp
        is_valid = time_diff < time_margin
        return is_valid

    def has_enough_battery(self) -> bool:
        self.current_battery_percenatage = self.data_parser.get_battery()
        has_enough = (self.current_battery_percenatage
                      >= self.min_battery_percenatage)
        return has_enough

    def is_connection_stable(self):
        if not self.data_parser.is_health:
            if self.prev_connection_stable:
                print("Ps4 controller connection is unstable!!")
                self.prev_connection_stable = False
            return self.prev_connection_stable

        if not self.prev_connection_stable:
            print("Ps4 controller connection is stable!!")
            self.prev_connection_stable = True
        return self.prev_connection_stable

    def listen(self) -> None:
        if not self.is_connection_stable():
            return

        for i in range(self.data_parser.get_numbuttons()):
            self.button_data[i] = self.data_parser.get_button(i)

        for i in range(self.data_parser.get_numaxes()):
            self.axis_data[i] = self.data_parser.get_axis(i)

        self.L_Xaxis = self.axis_data[0]
        self.L_Yaxis = self.axis_data[1]
        self.L_trigger = self.axis_data[2]

        self.R_Xaxis = self.axis_data[3]
        self.R_Yaxis = self.axis_data[4]
        self.R_trigger = self.axis_data[5]

        self.X_button = self.button_data[0]
        self.O_button = self.button_data[1]
        self.A_button = self.button_data[2]
        self.H_button = self.button_data[3]

        self.D_button = self.button_data[4]
        self.U_button = self.button_data[6]

        self.check_emergency_brake()

    def normalize_axis_value(self, value, margin: float = 0.1) -> float:
        # ignore small value(not triggered but has small value (ex. 0.019))
        if abs(value) < 0.1:
            normalized_value = 0.0
            return normalized_value
        normalized_value = (value -
                            margin * math.copysign(1, value)) / (1 - margin)
        return normalized_value

    def normalize_trigger_value(self, value: float) -> float:
        return (value + 1)

    def set_go_waypoint_condition(self, pub_condition: bool) -> None:
        self.pub_go_waypoint = pub_condition

    def set_stop_waypoint_condition(self, pub_condition: bool) -> None:
        self.pub_stop_waypoint = pub_condition

if __name__ == "__main__":

    ps4 = PS4Controller()

    while True:
        ps4.listen()
        # print(ps4.L_Xaxis, ps4.L_Yaxis, ps4.L_trigger)
        # print(ps4.R_Xaxis, ps4.R_Yaxis, ps4.R_trigger)
        # print(ps4._to_robot_action())
        time.sleep(0.001)
