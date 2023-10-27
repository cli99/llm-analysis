# Copyright 2023 Cheng Li
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def _num_to_string(num, precision=2, divisor=1024):
    if num is None:
        return "None"
    if num // divisor**4 > 0:
        return str(round(num / divisor**4, precision)) + " T"
    elif num // divisor**3 > 0:
        return str(round(num / divisor**3, precision)) + " G"
    elif num // divisor**2 > 0:
        return str(round(num / divisor**2, precision)) + " M"
    elif num // divisor > 0:
        return str(round(num / divisor, precision)) + " K"
    else:
        return str(num)


def _latency_to_string(latency_in_s, precision=2):
    if latency_in_s is None:
        return "None"
    day = 24 * 60 * 60
    hour = 60 * 60
    minute = 60
    ms = 1 / 1000
    us = 1 / 1000000
    if latency_in_s // day > 0:
        return str(round(latency_in_s / day, precision)) + " days"
    elif latency_in_s // hour > 0:
        return str(round(latency_in_s / hour, precision)) + " hours"
    elif latency_in_s // minute > 0:
        return str(round(latency_in_s / minute, precision)) + " minutes"
    elif latency_in_s > 1:
        return str(round(latency_in_s, precision)) + " s"
    elif latency_in_s > ms:
        return str(round(latency_in_s / ms, precision)) + " ms"
    else:
        return str(round(latency_in_s / us, precision)) + " us"


def within_range(val, target, tolerance):
    return abs(val - target) / target < tolerance
