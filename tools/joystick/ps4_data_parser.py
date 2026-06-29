import copy
import time
from collections import deque
from itertools import combinations
from threading import Thread
from typing import Dict, List, Set, Tuple

import hid


class PS4DataParser:

    def __init__(self, vendor: int, product: int) -> None:

        self.vendor = vendor
        self.product = product
        self.hid = hid.device()
        self.hid.open(self.vendor, self.product)
        self.hid.set_nonblocking(True)
        self.hid.get_feature_report(2, 100)

        self.button_number = 8  # X, O, A, D, DD, RR, UU, RR
        self.button_keys = ["X", "O", "A", "H"]
        button_values = [32, 64, 128, 16]
        button_items = dict(zip(self.button_keys, button_values))

        direction_keys = ["D", "R", "U", "L", "D,R", "R,U", "U,L", "L,D"]
        direction_values = [-4, -6, -8, -2, -5, -7, -1, -3]
        direction_items = dict(zip(direction_keys, direction_values))

        self.button_keys += direction_keys
        self.button_combination = self._make_combinations(
            button_items, direction_items)

        self.axis_number = 6  # Only consider L2, Lx,Ly, R2, Rx,Ry
        self.trigger_keys = ["L1", "R1", "L2", "R2", "SH", "OP", "LA", "RA"]
        self.trigger_numbers = 8
        trigger_values = [1, 2, 4, 8, 16, 32, 64, 128]
        trigger_items = dict(zip(self.trigger_keys, trigger_values))
        self.trigger_combination = self._make_combinations(trigger_items)

        self.worker_period = 0.001
        self.ps4_raw_data = []
        self.is_health = False
        window_size = 60
        self.valid_que = deque([True], maxlen=window_size)
        self.worker_is_running = True
        self.worker = Thread(target=self._check_health, args=(), daemon=True)
        self.worker.start()

    def _check_health(self) -> None:
        while self.worker_is_running:
            is_valid = self._is_data_valid()
            self.valid_que.append(is_valid)
            time.sleep(self.worker_period)
            if any(self.valid_que):
                self.is_health = True
                continue
            self.is_health = False

    def _is_data_valid(self) -> bool:
        ps4_raw_data = self.hid.read(32)

        # When ps4 is disconnected, it sends the same data.
        is_disconnected = self.ps4_raw_data == ps4_raw_data
        has_enough_data = len(ps4_raw_data) > 30 and len(ps4_raw_data) <= 34
        if ps4_raw_data:
            self.ps4_raw_data = ps4_raw_data

        if has_enough_data and not is_disconnected:
            return True
        return False

    def _make_combinations(
        self, combination_items: Dict[str, int], restrictions: dict = dict()
    ) -> Dict[int, Set[str]]:
        results = dict()
        results[0] = None
        el_keys = list(combination_items.keys())
        total_elments = copy.deepcopy(combination_items)
        total_elments.update(restrictions)
        for i in range(1, len(el_keys) + 1 + 1):
            if restrictions:
                for a_res in restrictions.keys():
                    target_elements = el_keys + [a_res]
                    for combination in combinations(target_elements, i):
                        results[sum(total_elments[key]
                                    for key in combination)] = combination
            else:
                for combination in combinations(el_keys, i):
                    results[sum(total_elments[key]
                                for key in combination)] = combination
        return results

    def get_axis(self, index: int) -> float:
        axises_data = self.get_axes()
        return axises_data[index]

    def get_axes(self) -> List[float]:
        # L_X, L_Y, L2, R_X, R_Y, R2 -> strength
        axes_data = [0. for _ in range(self.axis_number)]
        l_axis_strength = self._get_l_axis_strength()
        axes_data[:2] = l_axis_strength
        axes_data[2] = self._get_l2_strength()
        r_axis_strength = self._get_r_axis_strength()
        axes_data[3:5] = r_axis_strength
        axes_data[5] = self._get_r2_strength()
        return axes_data

    def _get_l_axis_strength(self) -> Tuple[float, float]:
        return self._get_strength(self.ps4_raw_data[1:3])

    def _get_l2_strength(self) -> float:
        return self.ps4_raw_data[8] / 255

    def _get_r_axis_strength(self) -> float:
        return self._get_strength(self.ps4_raw_data[3:5])

    def _get_strength(self, raw_data, default=128):
        x = (raw_data[0] - default) / default
        y = (raw_data[1] - default) / default
        return x, y

    def _get_r2_strength(self) -> float:
        return self.ps4_raw_data[9] / 255

    def get_battery(self) -> int:
        return self.ps4_raw_data[30] * 10

    def get_button(self, index: int) -> bool:
        # "X", "O", "A", "H", "D", "R", "U", "L"
        buttons_data = self.get_buttons()
        return buttons_data[index]

    def get_buttons(self, default: int = 8) -> List[bool]:
        return self._parse_combinations(self.ps4_raw_data[5] - default,
                                        self.button_number, self.button_keys,
                                        self.button_combination)

    def get_trigger(self, index: int) -> bool:
        # "L1", "R1", "L2", "R2", "SH", "OP", "LA", "RA"
        triggers_data = self.get_triggers()
        return triggers_data[index]

    def get_triggers(self) -> List[bool]:
        return self._parse_combinations(self.ps4_raw_data[6],
                                        self.trigger_numbers,
                                        self.trigger_keys,
                                        self.trigger_combination)

    def _parse_combinations(
            self, ps4_raw_data: int, button_number: int,
            button_keys: List[str],
            button_combination: Dict[int, Set[str]]) -> List[bool]:
        buttons = set()
        results = [False for _ in range(button_number)]
        combinations = button_combination[ps4_raw_data]
        if not combinations:
            return results
        for a_combination in combinations:
            for a_button in a_combination.split(","):
                buttons.add(a_button)
        for a_button in buttons:
            index = button_keys.index(a_button)
            results[index] = True
        return results

    def get_numaxes(self) -> int:
        return self.axis_number

    def get_numbuttons(self) -> int:
        return self.button_number

    def stop(self) -> None:
        self.worker_is_running = False
        self.hid.close()
        self.worker.join()


# if __name__ == "__main__":
#     ps4_health_checker = PS4DataParser(0x054C, 0x0ba0)
#     try:
#         ps4_health_checker.check_health()
#     except:
#         ps4_health_checker.stop()
