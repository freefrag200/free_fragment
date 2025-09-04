# FILE: main.py
import sys
import os
import json
import subprocess
import time
import requests
import copy
import base64
import csv
import tempfile
import threading
import multiprocessing
import concurrent.futures
import socket
import random
import statistics
from urllib.parse import urlparse, parse_qs
from queue import Empty
import io

import psutil
import qrcode
from PIL import Image

from PySide6.QtGui import (QImage, QTextCursor, QIcon, QKeySequence, QAction,
                           QPixmap, QColor, QFont, QDragEnterEvent, QDropEvent)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QProgressBar, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QSpinBox,
    QFileDialog, QMessageBox, QPlainTextEdit, QComboBox, QMenu, QStyle,
    QDialog, QDialogButtonBox, QGroupBox
)
from PySide6.QtCore import QThread, QObject, Signal, Qt, QByteArray

# --- Platform Specific Imports and Checks ---
IS_ADMIN = False
if sys.platform == "win32":
    import ctypes
    try:
        IS_ADMIN = ctypes.windll.shell32.IsUserAnAdmin()
    except:
        IS_ADMIN = False

# بررسی وابستگی‌ها
try:
    import socks
except ImportError:
    print("CRITICAL: PySocks library not found. Please run 'pip install PySocks'")
    sys.exit(1)

# --- ثابت‌ها ---
XRAY_EXECUTABLE = "xray.exe" if sys.platform == "win32" else "xray"
SETTINGS_FILE = "settings.json"
BEST_CONFIGS_FILE = "best_configs.txt"
BASE_CONFIG_TEMPLATE = {
  "log": {"loglevel": "warning"},
  "inbounds": [],
  "outbounds": [
    {},
    {"tag": "fragment", "protocol": "freedom", "settings": {"domainStrategy": "AsIs", "fragment": {}}},
    {"tag": "direct", "protocol": "freedom"},
    {"tag": "block", "protocol": "blackhole"}
  ],
  "routing": {"domainStrategy": "AsIs", "rules": [{"type": "field", "port": "0-65535", "outboundTag": "proxy"}]}
}
XRAY_CORE_URL_WINDOWS = "https://github.com/XTLS/Xray-core/releases/latest/download/Xray-windows-64.zip"


# --- توابع کمکی ---

def kill_xray_processes():
    """تمام فرآیندهای xray.exe در حال اجرا را می‌بندد."""
    if sys.platform != "win32":
        return
    try:
        subprocess.run(['taskkill', '/F', '/IM', XRAY_EXECUTABLE], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL, 
                         check=False)
    except FileNotFoundError:
        pass # taskkill might not be in PATH
    except Exception:
        pass # Ignore other errors

class DragDropLineEdit(QLineEdit):
    """A QLineEdit that accepts drag and drop for file paths."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() and event.mimeData().urls()[0].isLocalFile():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.setText(file_path)

class XrayProcessManager:
    """یک context manager برای مدیریت ایمن فرآیندها و فایل‌های موقت Xray."""
    def __init__(self, xray_path, config_data, socks_port=None):
        self.xray_path = xray_path
        self.config_data = config_data
        self.socks_port = socks_port
        self.process = None
        self.temp_config_file = None

    def __enter__(self):
        self.temp_config_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8')
        json.dump(self.config_data, self.temp_config_file)
        self.temp_config_file.close()

        command = [self.xray_path, "-c", self.temp_config_file.name]
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creation_flags)
        
        if self.socks_port: # Only check port if it's a SOCKS inbound
            is_ready = False
            for _ in range(15): # Try for ~3 seconds
                if self.process.poll() is not None:
                    break 
                try:
                    with socket.create_connection(("127.0.0.1", self.socks_port), timeout=0.2) as s:
                        is_ready = True
                        break
                except (socket.timeout, ConnectionRefusedError):
                    time.sleep(0.2)
            
            if not is_ready:
                err_output = self.process.stderr.read().decode('utf-8', errors='ignore')
                self.process.terminate()
                raise RuntimeError(f"فرآیند Xray نتوانست روی پورت {self.socks_port} اجرا شود. خطا:\n{err_output}")
        else: # For other modes, just a small wait is enough
            time.sleep(1.5)
            if self.process.poll() is not None:
                err_output = self.process.stderr.read().decode('utf-8', errors='ignore')
                self.process.terminate()
                raise RuntimeError(f"فرآیند Xray نتوانست اجرا شود. خطا:\n{err_output}")
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        
        if self.temp_config_file and os.path.exists(self.temp_config_file.name):
            try:
                os.remove(self.temp_config_file.name)
            except OSError:
                pass

class SubTestSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("تنظیمات تست اشتراک")
        layout = QGridLayout(self)

        layout.addWidget(QLabel("حداکثر تعداد ترد:"), 0, 0)
        self.threads_combo = QComboBox()
        self.threads_combo.addItems(["10", "20", "50", "100"])
        self.threads_combo.setCurrentText("100")
        layout.addWidget(self.threads_combo, 0, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box, 1, 0, 1, 2)

    def get_settings(self):
        return int(self.threads_combo.currentText())

    @staticmethod
    def run(parent=None):
        dialog = SubTestSettingsDialog(parent)
        if dialog.exec() == QDialog.Accepted:
            return dialog.get_settings()
        return None

# --- توابع解析 و کمکی برای تستر اشتراک ---
def parse_trojan(uri_str):
    try:
        parsed = urlparse(uri_str)
        password = parsed.username
        server = parsed.hostname
        port = parsed.port
        query = parse_qs(parsed.query)
        sni = query.get("sni", [server])[0]
        network = query.get("type", ["tcp"])[0]
        
        stream_settings = {
            "network": network,
            "security": "tls",
            "tlsSettings": { "serverName": sni }
        }
        if network == "ws":
            stream_settings["wsSettings"] = {
                "path": query.get("path", ["/"])[0],
                "headers": { "Host": query.get("host", [server])[0] }
            }
        
        outbound = {
            "protocol": "trojan",
            "tag": "proxy",
            "settings": { "servers": [{"address": server, "port": port, "password": password}] },
            "streamSettings": stream_settings
        }
        name = parsed.fragment or f"trojan_{server}"
        return outbound, name, uri_str
    except Exception:
        return None, None, None

def parse_shadowsocks(uri_str):
    try:
        parsed_url = urlparse(uri_str)
        server = parsed_url.hostname
        port = parsed_url.port
        remark = parsed_url.fragment or f"ss_{server}"
        
        encoded_part = uri_str.split('//')[1].split('@')[0]
        user_info = base64.b64decode(encoded_part + '=' * (-len(encoded_part) % 4)).decode('utf-8')
        
        parts = user_info.split(':')
        if len(parts) != 2: return None, None, None
        method, password = parts
        
        outbound = {
            "protocol": "shadowsocks",
            "tag": "proxy",
            "settings": { "servers": [{"address": server, "port": port, "method": method, "password": password}] }
        }
        return outbound, remark, uri_str
    except Exception:
        return None, None, None

def universal_parse_proxy_link(link: str):
    try:
        if link.startswith("vmess://"):
            base64_str = link[8:]
            padded_base64 = base64_str + '=' * (-len(base64_str) % 4)
            decoded_json = base64.b64decode(padded_base64).decode('utf-8')
            vmess_data = json.loads(decoded_json)
            outbound = {"protocol": "vmess", "tag": "proxy", "settings": {"vnext": [{"address": vmess_data.get("add"), "port": int(vmess_data.get("port")), "users": [{"id": vmess_data.get("id"), "alterId": int(vmess_data.get("aid", 0)), "security": vmess_data.get("scy", "auto")}]}]}, "streamSettings": {"network": vmess_data.get("net"), "security": vmess_data.get("tls"), "wsSettings": {"path": vmess_data.get("path"), "headers": {"Host": vmess_data.get("host")}}}}
            if vmess_data.get("tls") == "tls": outbound["streamSettings"]["tlsSettings"] = {"serverName": vmess_data.get("sni") or vmess_data.get("host")}
            return outbound, vmess_data.get("ps", "Unnamed VMESS"), link
        elif link.startswith("vless://"):
            parsed_url = urlparse(link)
            params = parse_qs(parsed_url.query)
            outbound = {"protocol": "vless", "tag": "proxy", "settings": {"vnext": [{"address": parsed_url.hostname, "port": parsed_url.port, "users": [{"id": parsed_url.username, "encryption": "none"}]}]}, "streamSettings": {"network": params.get("type", [None])[0], "security": params.get("security", [None])[0]}}
            if params.get("security", ["none"])[0] == "tls": outbound["streamSettings"]["tlsSettings"] = {"serverName": params.get("sni", [parsed_url.hostname])[0]}
            if params.get("type", ["tcp"])[0] == "ws": outbound["streamSettings"]["wsSettings"] = {"path": params.get("path", ["/"])[0], "headers": {"Host": params.get("host", [parsed_url.hostname])[0]}}
            if "streamSettings" not in outbound: outbound["streamSettings"] = {}
            outbound["streamSettings"]["sockopt"] = {"dialerProxy": "fragment"}
            return outbound, parsed_url.fragment or "Unnamed VLESS", link
        elif link.startswith("ss://"):
            return parse_shadowsocks(link)
        elif link.startswith("trojan://"):
            return parse_trojan(link)
    except Exception: return None, None, None
    return None, None, None

def universal_parse_protocol(uri, socks_port=10808):
    outbound, _, _ = universal_parse_proxy_link(uri)
    if not outbound: return None
    return {
        "log": {"loglevel": "warning"},
        "inbounds": [{"port": socks_port, "protocol": "socks", "listen": "127.0.0.1", "settings": {"udp": True}}],
        "outbounds": [outbound, {"protocol": "freedom", "tag": "direct"}, {"protocol": "blackhole", "tag": "block"}],
        "routing": {"rules": [{"type": "field", "ip": ["geoip:private"], "outboundTag": "direct"}]}
    }

def get_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def generate_free_packets():
    packets = ["tlshello"]; packets.extend([f"1-{i}" for i in range(1, 11)]); return packets
def generate_free_length():
    return [f"{i}-{min(i + 4, 500)}" for i in range(1, 501, 5)]
def generate_free_interval():
    return [f"{i}-{min(i + 4, 50)}" for i in range(1, 51, 5)]

class DownloadWorker(QObject):
    log_message = Signal(str, str); finished = Signal(bool, str)
    def __init__(self, url, save_path, file_id): super().__init__(); self.url = url; self.save_path = save_path; self.file_id = file_id
    def run(self):
        try:
            self.log_message.emit("INFO", f"شروع دانلود {self.file_id} به مسیر: {self.save_path}")
            response = requests.get(self.url, stream=True, timeout=30); response.raise_for_status()
            with open(self.save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
            self.log_message.emit("INFO", f"فایل {self.file_id} با موفقیت دانلود/به‌روزرسانی شد.")
            self.finished.emit(True, self.file_id)
        except requests.exceptions.RequestException as e: self.log_message.emit("ERROR", f"خطا در دانلود {self.file_id}: {e}"); self.finished.emit(False, self.file_id)
        except Exception as e: self.log_message.emit("ERROR", f"خطای نامشخص در دانلود {self.file_id}: {e}"); self.finished.emit(False, self.file_id)


class FetchWorker(QObject):
    proxies_fetched = Signal(dict); fetch_failed = Signal(str); finished = Signal()
    def __init__(self, url): super().__init__(); self.url = url
    def run(self):
        try:
            response = requests.get(self.url, timeout=10); response.raise_for_status()
            content = response.text
            if response.encoding != 'utf-8':
                 try:
                     content = base64.b64decode(response.content).decode('utf-8')
                 except:
                     pass # keep original if fails
            links = content.splitlines()
            proxies = {}
            for link in links:
                if link.strip():
                    outbound, name, original_link = universal_parse_proxy_link(link.strip())
                    if outbound and name: proxies[f"{name} ({len(proxies)})"] = (outbound, original_link)
            if not proxies: self.fetch_failed.emit("هیچ پروکسی معتبری در لینک یافت نشد."); return
            self.proxies_fetched.emit(proxies)
        except requests.exceptions.RequestException as e: self.fetch_failed.emit(f"خطا در دریافت یا پردازش لینک: {e}")
        except Exception as e: self.fetch_failed.emit(f"خطای نامشخص: {e}")
        finally: self.finished.emit()

class ManagerWorker(QObject):
    log_message = Signal(str, str)
    progress_updated = Signal(int, int)
    result_found = Signal(dict, str)
    finished = Signal()

    def __init__(self, settings, base_config):
        super().__init__()
        self.settings = settings
        self.base_config = base_config
        self._is_running = True
        self.param_scores = {
            'packets': {p: 1 for p in self.settings['packets_list']},
            'length': {l: 1 for l in self.settings['length_list']},
            'interval': {i: 1 for i in self.settings['interval_list']}
        }

    def run(self):
        packets_list, length_list, interval_list = self.settings['packets_list'], self.settings['length_list'], self.settings['interval_list']
        total_tasks = len(packets_list) * len(length_list) * len(interval_list)
        self.log_message.emit("INFO", f"تعداد کل ترکیبات برای تست: {total_tasks}")
        self.log_message.emit("INFO", f"استفاده از {self.settings['num_workers']} هسته CPU برای پردازش هوشمند...")
        
        manager = multiprocessing.Manager()
        task_queue = manager.Queue()
        result_queue = manager.Queue()

        processes = [multiprocessing.Process(target=self.ping_test_worker, args=(task_queue, result_queue)) for _ in range(self.settings['num_workers'])]
        for p in processes: p.start()

        completed_tasks = 0
        for i in range(total_tasks):
            if not self._is_running: break
            params = self._get_next_params(i)
            task_queue.put((i, params, self.settings, self.settings['original_config_str'], self.base_config))

        while completed_tasks < total_tasks and self._is_running:
            try:
                result_data = result_queue.get(timeout=1)
                if self._is_running:
                    self._process_result(result_data)
                    completed_tasks += 1
                    self.progress_updated.emit(completed_tasks, total_tasks)
            except Empty:
                if not any(p.is_alive() for p in processes) and task_queue.empty():
                    break
                continue

        for _ in processes:
            try: task_queue.put(None)
            except Exception: pass
        for p in processes: p.join(timeout=5); p.kill() if p.is_alive() else None
        manager.shutdown()
        if self._is_running: self.finished.emit()

    def _get_next_params(self, task_index):
        if task_index < self.settings['num_workers'] or random.random() < 0.20:
            return {
                "packets": random.choice(self.settings['packets_list']),
                "length": random.choice(self.settings['length_list']),
                "interval": random.choice(self.settings['interval_list']),
            }
        else:
            return {
                "packets": self._get_weighted_random('packets'),
                "length": self._get_weighted_random('length'),
                "interval": self._get_weighted_random('interval'),
            }

    def _get_weighted_random(self, param_type):
        population = list(self.param_scores[param_type].keys())
        weights = list(self.param_scores[param_type].values())
        return random.choices(population, weights, k=1)[0]
    
    def _process_result(self, result_data):
        params = result_data['params']
        original_config_str = result_data['original_config_str']
        
        if result_data['avg_latency'] is not None:
            self.log_message.emit("INFO", f"موفق! تأخیر: {result_data['avg_latency']:.2f}ms, سرعت: {result_data['speed']:.2f} Mbps, پایداری: {result_data['success_rate']*100:.0f}%, جیتر: {result_data['jitter']:.2f}ms, تنظیمات: {params}")
            self.result_found.emit(result_data, original_config_str)

            score = result_data['score']
            if score > 0:
                self.param_scores['packets'][params['packets']] += score
                self.param_scores['length'][params['length']] += score
                self.param_scores['interval'][params['interval']] += score
        else:
            self.log_message.emit("DEBUG", f"ناموفق. تنظیمات: {params}")

    def stop(self):
        if self._is_running: self._is_running = False; self.log_message.emit("INFO", "ارسال سیگنال توقف..."); self.finished.emit()
    
    @staticmethod
    def ping_test_worker(task_queue, result_queue):
        while True:
            try:
                task = task_queue.get()
                if task is None: break
            except (Empty, EOFError): continue
            
            index, params, settings, original_config_str, base_config_for_worker = task
            config_data = copy.deepcopy(base_config_for_worker)
            fragment_outbound = next((o for o in config_data.get("outbounds", []) if o.get("tag") == "fragment"), None)
            if fragment_outbound:
                if 'settings' not in fragment_outbound: fragment_outbound['settings'] = {}
                fragment_outbound['settings']['fragment'] = params

            for i, inbound in enumerate(config_data['inbounds']):
                new_port = 11000 + (os.getpid() % 1000) * 10 + i
                inbound['port'] = new_port
            
            socks_port = next((ib['port'] for ib in config_data.get('inbounds', []) if ib.get('protocol') == 'socks'), None)
            if not socks_port:
                result_queue.put({"params": params, "original_config_str": original_config_str, "avg_latency": None}); continue

            try:
                with XrayProcessManager(settings['xray_path'], config_data, socks_port=socks_port):
                    proxy = f"socks5://127.0.0.1:{socks_port}"; proxies = {"http": proxy, "https": proxy}
                    
                    latencies = []
                    num_pings = 3
                    for _ in range(num_pings):
                        try:
                            start_time = time.perf_counter()
                            requests.head(settings['urls'][0], proxies=proxies, timeout=settings['timeout'], headers={'User-Agent': 'Mozilla/5.0'})
                            latency = (time.perf_counter() - start_time) * 1000
                            latencies.append(latency)
                        except requests.exceptions.RequestException:
                            pass 
                    
                    if not latencies:
                        result_queue.put({"params": params, "original_config_str": original_config_str, "avg_latency": None}); continue
                    
                    avg_latency = sum(latencies) / len(latencies)
                    jitter = statistics.stdev(latencies) if len(latencies) > 1 else 0
                    success_rate = len(latencies) / num_pings
                    speed = -1

                    if success_rate > 0.5:
                        try:
                            speed_url = 'http://speed.cloudflare.com/__down?bytes=1000000'
                            start_time = time.time()
                            response = requests.get(speed_url, proxies=proxies, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                            duration = time.time() - start_time
                            if duration > 0 and response.status_code == 200:
                                speed = (len(response.content) / 1024 / 1024 * 8) / duration
                        except requests.exceptions.RequestException:
                            speed = -1
                    
                    score = (speed * 8 + (success_rate * 100)) - (avg_latency) - (jitter * 2)
                    
                    result_data = {
                        "params": params, "original_config_str": original_config_str,
                        "avg_latency": avg_latency, "speed": speed, "jitter": jitter,
                        "success_rate": success_rate, "score": score
                    }
                    result_queue.put(result_data)

            except (RuntimeError, Exception) as e:
                result_queue.put({"params": params, "original_config_str": original_config_str, "avg_latency": None})


class SubscriptionTesterWorkerV2(QObject):
    log_message = Signal(str, str)
    progress_updated = Signal(int, int, int)
    result_found = Signal(str, float, str, str, str)
    finished = Signal(list)

    def __init__(self, settings, configs_to_test, save_mode='merge'):
        super().__init__()
        self.settings = settings
        self.configs_to_test = configs_to_test
        self.save_mode = save_mode
        self.stop_event = threading.Event()

    def run(self):
        working_results = []
        try:
            total_configs = len(self.configs_to_test)
            tested_configs = 0
            working_configs_count = 0
            self.progress_updated.emit(total_configs, tested_configs, working_configs_count)
            self.log_message.emit("INFO", f"شروع تست {total_configs} کانفیگ با {self.settings['num_workers']} ترد...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.settings['num_workers']) as executor:
                futures = {executor.submit(self._measure_latency, config): config for config in self.configs_to_test}
                for future in concurrent.futures.as_completed(futures):
                    if self.stop_event.is_set():
                        for f in futures: f.cancel()
                        break
                    
                    uri, latency, protocol, server, port = future.result()
                    tested_configs += 1
                    if latency != float('inf'):
                        working_configs_count += 1
                        working_results.append((uri, latency, protocol, server, port))
                        self.result_found.emit(uri, latency, protocol, server, port)
                    
                    self.progress_updated.emit(total_configs, tested_configs, working_configs_count)

            if not self.stop_event.is_set():
                self.log_message.emit("INFO", f"تست کامل شد. {working_configs_count} کانفیگ سالم یافت شد.")
                self._update_best_configs_file(working_results)

        except Exception as e:
            self.log_message.emit("ERROR", f"خطا در کارگر تستر اشتراک: {e}")
        finally:
            self.finished.emit(working_results)

    def _update_best_configs_file(self, new_working_results):
        try:
            new_working_uris = {res[0] for res in new_working_results}
            final_uris = set()

            if self.save_mode == 'merge':
                if os.path.exists(BEST_CONFIGS_FILE):
                    with open(BEST_CONFIGS_FILE, 'r', encoding='utf-8') as f:
                        existing_uris = {line.strip() for line in f if line.strip()}
                    final_uris.update(existing_uris)
                final_uris.update(new_working_uris)
            elif self.save_mode == 'overwrite':
                final_uris = new_working_uris
            else: # test_only mode
                return

            with open(BEST_CONFIGS_FILE, 'w', encoding='utf-8') as f:
                for u in final_uris: f.write(f"{u}\n")
            self.log_message.emit("INFO", f"فایل {BEST_CONFIGS_FILE} به‌روزرسانی شد.")
        except Exception as e:
            self.log_message.emit("ERROR", f"خطا در ذخیره فایل بهترین کانفیگ‌ها: {e}")

    def _measure_latency(self, config_uri):
        if self.stop_event.is_set(): return (config_uri, float('inf'), "N/A", "N/A", "N/A")
        
        protocol, server, port = self._parse_config_info(config_uri)
        socks_port = get_available_port()
        if not socks_port:
             return (config_uri, float('inf'), protocol, server, port)

        try:
            config_json = universal_parse_protocol(config_uri, socks_port=socks_port)
            if not config_json:
                return (config_uri, float('inf'), protocol, server, port)

            with XrayProcessManager(self.settings['xray_path'], config_json, socks_port=socks_port):
                proxies = {'http': f'socks5://127.0.0.1:{socks_port}', 'https': f'socks5://127.0.0.1:{socks_port}'}
                start_time = time.perf_counter()
                response = requests.head(
                    "https://old-queen-f906.mynameissajjad.workers.dev/login",
                    proxies=proxies,
                    timeout=8,
                    headers={'Cache-Control': 'no-cache', 'Connection': 'close'}
                )
                response.raise_for_status()
                latency = (time.perf_counter() - start_time) * 1000
                return (config_uri, latency, protocol, server, port)
        except (requests.RequestException, RuntimeError, ValueError):
            return (config_uri, float('inf'), protocol, server, port)
        except Exception:
            return (config_uri, float('inf'), protocol, server, port)

    def _parse_config_info(self, uri):
        try:
            protocol_map = {
                "vmess://": "vmess",
                "vless://": "vless",
                "trojan://": "trojan",
                "ss://": "shadowsocks"
            }
            for prefix, protocol in protocol_map.items():
                if uri.startswith(prefix):
                    parsed = urlparse(uri)
                    if prefix == "vmess://":
                        data = json.loads(base64.b64decode(uri[8:] + '==').decode('utf-8'))
                        return protocol, data.get("add", "N/A"), str(data.get("port", "N/A"))
                    return protocol, parsed.hostname, str(parsed.port)
        except Exception: pass
        return "unknown", "N/A", "N/A"

    def stop(self):
        self.stop_event.set()


class MainWindow(QMainWindow):
    LOG_COLORS = {"DEBUG": QColor("#888888"), "INFO": QColor("#FFFFFF"), "ERROR": QColor("#FF5555")}
    def __init__(self):
        super().__init__()
        kill_xray_processes() # Kill any leftover processes on startup
        self.DEFAULT_SUBSCRIPTIONS = {
            "barry-far": "https://raw.githubusercontent.com/barry-far/V2ray-Config/main/All_Configs_Sub.txt",
            "SoliSpirit": "https://raw.githubusercontent.com/SoliSpirit/v2ray-configs/main/all_configs.txt",
            "PSG | MIX": "https://raw.githubusercontent.com/itsyebekhe/PSG/main/subscriptions/xray/base64/mix"
        }
        self.URL_PROFILES = {
            "YouTube": "https://www.youtube.com/",
            "Twitter": "https://www.twitter.com/",
            "Facebook": "https://www.facebook.com/",
            "Telegram": "https://www.telegram.org/",
            "Cloudflare": "https://speed.cloudflare.com/",
            "سفارشی": ""
        }
        self.subscription_profiles = {}
        self.base_title = "Config Tester & Analyzer"
        self.setWindowTitle(self.base_title)
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.thread = None; self.worker = None; self.fetched_proxies = {}; self.base_config_for_saving = None
        self.best_latency = float('inf'); self.best_params = None; self.update_xray_action = None; self.download_thread = None; self.fetch_thread = None
        self.is_sub_testing = False
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        self.main_widget = QWidget(); self.setCentralWidget(self.main_widget)
        self.setup_modern_dark_theme_v2()
        main_layout = QVBoxLayout(self.main_widget)
        self.create_menu_bar()
        self.tabs = QTabWidget(); main_layout.addWidget(self.tabs)
        self.create_settings_ui(); self.create_results_tab(); self.create_subscription_tester_tab()
        self.progress_bar = QProgressBar(); main_layout.addWidget(self.progress_bar)

    def setup_modern_dark_theme_v2(self):
        QApplication.instance().setStyle("Fusion")
        self.setStyleSheet("""
            QWidget { background-color: #202124; color: #E8EAED; border: none; font-family: Segoe UI; font-size: 10pt; }
            QMainWindow { background-color: #202124; }
            QTabWidget::pane { border-top: 2px solid #323639; }
            QTabBar::tab { background: #202124; border: 1px solid #202124; padding: 8px 20px; border-bottom: none; }
            QTabBar::tab:hover { background: #323639; }
            QTabBar::tab:selected { background: #323639; border-bottom: 2px solid #8AB4F8; color: #8AB4F8; }
            QTableWidget { gridline-color: #323639; background-color: #202124; }
            QHeaderView::section { background-color: #323639; padding: 5px; border: none; }
            QPushButton { background-color: #303134; border: 1px solid #5F6368; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background-color: #3c4043; }
            QPushButton:pressed { background-color: #5E6165; }
            QPushButton:disabled { background-color: #282828; color: #888888; }
            QComboBox, QSpinBox, QLineEdit, QPlainTextEdit { background-color: #303134; border: 1px solid #5F6368; padding: 5px; border-radius: 4px; }
            QComboBox::drop-down { border: none; }
            QProgressBar { border: 1px solid #5F6368; border-radius: 4px; text-align: center; color: white; }
            QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4186F5, stop:1 #0059C1); border-radius: 4px; }
            QMenu { background-color: #3c4043; border: 1px solid #5F6368; }
            QMenu::item:selected { background-color: #4a5055; }
            QTableWidget::item:selected { background-color: #4a6984; }
            QTableWidget::item { padding: 5px; }
            QGroupBox { border: 1px solid #5F6368; border-radius: 4px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }
        """)

    def create_menu_bar(self):
        menu_bar = self.menuBar()
        tools_menu = menu_bar.addMenu("ابزارها")
        self.update_xray_action = QAction("به‌روزرسانی هسته Xray", self)
        self.update_xray_action.triggered.connect(self.start_xray_update)
        tools_menu.addAction(self.update_xray_action)

    def create_settings_ui(self):
        settings_tab_widget = QWidget(); settings_layout = QVBoxLayout(settings_tab_widget)
        top_controls_layout = QHBoxLayout()
        self.profile_combo = QComboBox(); self.profile_combo.addItems(["free", "سفارشی"])
        self.profile_combo.currentTextChanged.connect(self.apply_test_profile)
        top_controls_layout.addWidget(QLabel("پروفایل تست:")); top_controls_layout.addWidget(self.profile_combo); top_controls_layout.addStretch()
        settings_layout.addLayout(top_controls_layout)
        
        self.input_tabs = QTabWidget(); settings_layout.addWidget(self.input_tabs)
        self.create_file_input_tab(); self.create_sub_input_tab()
        
        common_settings_grid = QGridLayout()
        
        common_settings_grid.addWidget(QLabel("URL تست پایداری:"), 0, 0)
        url_layout = QHBoxLayout()
        self.url_combo = QComboBox()
        self.url_combo.addItems(self.URL_PROFILES.keys())
        self.url_combo.currentTextChanged.connect(self.on_url_profile_changed)
        url_layout.addWidget(self.url_combo)
        
        self.custom_url_edit = QLineEdit()
        self.custom_url_edit.setPlaceholderText("آدرس سفارشی را وارد کنید...")
        self.custom_url_edit.setVisible(False)
        url_layout.addWidget(self.custom_url_edit)
        common_settings_grid.addLayout(url_layout, 0, 1)

        common_settings_grid.addWidget(QLabel("زمان وقفه (ثانیه):"), 1, 0); self.timeout_spin = QSpinBox(); self.timeout_spin.setRange(1, 60); common_settings_grid.addWidget(self.timeout_spin, 1, 1)
        common_settings_grid.addWidget(QLabel("تعداد هسته CPU برای استفاده:"), 2, 0); self.cpu_spin = QSpinBox(); self.cpu_spin.setRange(1, multiprocessing.cpu_count()); common_settings_grid.addWidget(self.cpu_spin, 2, 1)
        settings_layout.addLayout(common_settings_grid)

        values_group = QGroupBox("مقادیر تست (Values)")
        values_layout = QHBoxLayout(values_group)
        self.packets_edit = self.create_fragment_box(values_layout, "Packets")
        self.length_edit = self.create_fragment_box(values_layout, "Length")
        self.interval_edit = self.create_fragment_box(values_layout, "Interval")
        settings_layout.addWidget(values_group)
        
        control_group = QHBoxLayout()
        self.start_button = QPushButton(" شروع اسکن هوشمند"); self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay)); self.start_button.clicked.connect(self.start_scan)
        self.stop_button = QPushButton(" توقف اسکن"); self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop)); self.stop_button.clicked.connect(self.stop_scan); self.stop_button.setEnabled(False)
        self.save_best_button = QPushButton(" ذخیره بهترین"); self.save_best_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton)); self.save_best_button.clicked.connect(self.save_best_config); self.save_best_button.setEnabled(False)
        control_group.addWidget(self.start_button); control_group.addWidget(self.stop_button); control_group.addWidget(self.save_best_button)
        settings_layout.addLayout(control_group)
        self.tabs.addTab(settings_tab_widget, "بهینه‌ساز فرگمنت")

    def create_file_input_tab(self):
        file_tab = QWidget(); layout = QGridLayout(file_tab)
        layout.addWidget(QLabel("مسیر فایل Xray:"), 0, 0); self.xray_path_edit = DragDropLineEdit(os.path.abspath(XRAY_EXECUTABLE)); layout.addWidget(self.xray_path_edit, 0, 1)
        browse_xray_btn = QPushButton(); browse_xray_btn.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon)); browse_xray_btn.clicked.connect(lambda: self.browse_path(self.xray_path_edit, "انتخاب فایل Xray")); layout.addWidget(browse_xray_btn, 0, 2)
        layout.addWidget(QLabel("مسیر فایل Config:"), 1, 0); self.config_path_edit = DragDropLineEdit(os.path.abspath("config.json")); layout.addWidget(self.config_path_edit, 1, 1)
        browse_config_btn = QPushButton(); browse_config_btn.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon)); browse_config_btn.clicked.connect(lambda: self.browse_path(self.config_path_edit, "انتخاب کانفیگ", "JSON Files (*.json)")); layout.addWidget(browse_config_btn, 1, 2)
        self.input_tabs.addTab(file_tab, "از فایل")

    def create_sub_input_tab(self):
        sub_tab = QWidget(); layout = QVBoxLayout(sub_tab)
        sub_layout = QHBoxLayout()
        self.sub_url_edit = QLineEdit(); self.sub_url_edit.setPlaceholderText("لینک اشتراک خود را اینجا وارد کنید...")
        self.fetch_button = QPushButton("دریافت پروکسی‌ها"); self.fetch_button.clicked.connect(self.fetch_proxies)
        sub_layout.addWidget(self.sub_url_edit); sub_layout.addWidget(self.fetch_button)
        layout.addLayout(sub_layout)
        self.proxy_combo = QComboBox(); self.proxy_combo.setEnabled(False)
        layout.addWidget(self.proxy_combo); self.input_tabs.addTab(sub_tab, "از لینک اشتراک")

    def create_fragment_box(self, layout, title):
        box = QVBoxLayout(); box.addWidget(QLabel(f"{title}:")); text_edit = QPlainTextEdit(); text_edit.setFont(QFont("Courier New", 10)); box.addWidget(text_edit); layout.addLayout(box); return text_edit

    def create_results_tab(self):
        results_widget = QWidget(); layout = QVBoxLayout(results_widget)
        top_info_layout = QHBoxLayout(); self.best_ping_label = QLabel("بهترین پینگ: N/A"); self.best_ping_label.setStyleSheet("font-size: 12pt; color: #8AB4F8;"); top_info_layout.addWidget(self.best_ping_label); top_info_layout.addStretch()
        copy_log_button = QPushButton("کپی کردن لاگ"); copy_log_button.clicked.connect(self.copy_log_to_clipboard)
        self.export_csv_button = QPushButton("خروجی CSV"); self.export_csv_button.clicked.connect(self.export_results_to_csv); self.export_csv_button.setEnabled(False)
        top_info_layout.addWidget(copy_log_button); top_info_layout.addWidget(self.export_csv_button); layout.addLayout(top_info_layout)
        self.log_area = QPlainTextEdit(); self.log_area.setReadOnly(True); self.log_area.setFont(QFont("Courier New", 9)); layout.addWidget(self.log_area)
        self.results_table = QTableWidget(); self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels(["امتیاز", "تأخیر (ms)", "سرعت (Mbps)", "پایداری (%)", "جیتر (ms)", "Packets", "Length", "Interval"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); self.results_table.setSortingEnabled(True); layout.addWidget(self.results_table)
        self.setup_table_context_menu(self.results_table, self.show_qr_code_for_fragment)
        self.tabs.addTab(results_widget, "نتایج فرگمنت")

    def create_subscription_tester_tab(self):
        sub_tester_widget = QWidget(); layout = QVBoxLayout(sub_tester_widget)

        sub_management_layout = QGridLayout()
        self.sub_links_combo = QComboBox(); self.sub_links_combo.setPlaceholderText("پروفایل اشتراک را انتخاب کنید")
        self.new_sub_name_edit = QLineEdit(); self.new_sub_name_edit.setPlaceholderText("نام پروفایل اشتراک")
        self.new_sub_link_edit = QLineEdit(); self.new_sub_link_edit.setPlaceholderText("لینک اشتراک جدید")
        self.add_sub_button = QPushButton("افزودن پروفایل"); self.add_sub_button.clicked.connect(self.add_subscription_profile)
        self.remove_sub_button = QPushButton("حذف پروفایل"); self.remove_sub_button.clicked.connect(self.remove_subscription_profile)
        sub_management_layout.addWidget(self.sub_links_combo, 0, 0, 1, 2)
        sub_management_layout.addWidget(self.remove_sub_button, 0, 2)
        sub_management_layout.addWidget(self.new_sub_name_edit, 1, 0)
        sub_management_layout.addWidget(self.new_sub_link_edit, 1, 1)
        sub_management_layout.addWidget(self.add_sub_button, 1, 2)
        layout.addLayout(sub_management_layout)
        
        test_controls_layout = QHBoxLayout()
        self.sub_fetch_button = QPushButton("دریافت و تست از پروفایل منتخب"); self.sub_fetch_button.clicked.connect(self.start_sub_test_from_url)
        self.sub_stop_button = QPushButton("توقف کامل"); self.sub_stop_button.clicked.connect(self.stop_sub_test); self.sub_stop_button.setEnabled(False)
        self.copy_all_working_button = QPushButton("کپی همه کانفیگ‌های سالم"); self.copy_all_working_button.clicked.connect(self.copy_all_working_configs)
        test_controls_layout.addWidget(self.sub_fetch_button)
        test_controls_layout.addWidget(self.copy_all_working_button)
        test_controls_layout.addWidget(self.sub_stop_button)
        layout.addLayout(test_controls_layout)
        
        counters_layout = QHBoxLayout()
        self.sub_total_label = QLabel("کل: 0"); self.sub_tested_label = QLabel("تست‌شده: 0"); self.sub_working_label = QLabel("سالم: 0")
        counters_layout.addWidget(self.sub_total_label); counters_layout.addWidget(self.sub_tested_label); counters_layout.addWidget(self.sub_working_label)
        counters_layout.addStretch(); layout.addLayout(counters_layout)

        self.sub_test_results_table = QTableWidget(); self.sub_test_results_table.setColumnCount(5)
        self.sub_test_results_table.setHorizontalHeaderLabels(["تأخیر (ms)", "پروتکل", "سرور", "پورت", "کانفیگ"])
        self.sub_test_results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents); self.sub_test_results_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.sub_test_results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows); self.sub_test_results_table.setSortingEnabled(True)
        self.sub_test_results_table.setContextMenuPolicy(Qt.CustomContextMenu); self.sub_test_results_table.customContextMenuRequested.connect(self.show_sub_table_context_menu)
        layout.addWidget(self.sub_test_results_table)
        
        copy_action = QAction("Copy", self); copy_action.setShortcut(QKeySequence.StandardKey.Copy); copy_action.triggered.connect(self.copy_sub_config); self.sub_test_results_table.addAction(copy_action)
        delete_action = QAction("Delete", self); delete_action.setShortcut(QKeySequence.StandardKey.Delete); delete_action.triggered.connect(self.delete_sub_config); self.sub_test_results_table.addAction(delete_action)
        self.tabs.addTab(sub_tester_widget, "تستر اشتراک")
        paste_action = QAction("Paste", self); paste_action.setShortcut(QKeySequence.StandardKey.Paste); paste_action.triggered.connect(self.paste_and_test_configs); self.addAction(paste_action)

    def apply_test_profile(self, profile_name):
        is_custom = profile_name == "سفارشی"
        self.packets_edit.setReadOnly(not is_custom); self.length_edit.setReadOnly(not is_custom); self.interval_edit.setReadOnly(not is_custom)
        if profile_name == "free": 
            self.packets_edit.setPlainText("\n".join(generate_free_packets()))
            self.length_edit.setPlainText("\n".join(generate_free_length()))
            self.interval_edit.setPlainText("\n".join(generate_free_interval()))
        elif profile_name == "سفارشی": 
            pass

    def on_url_profile_changed(self, profile_name):
        is_custom = profile_name == "سفارشی"
        self.custom_url_edit.setVisible(is_custom)

    def start_xray_update(self):
        if self.download_thread and self.download_thread.isRunning():
            QMessageBox.warning(self, "توجه", "یک عملیات دانلود دیگر در حال انجام است.")
            return

        xray_dir = os.path.dirname(os.path.abspath(self.xray_path_edit.text()))
        if not os.path.isdir(xray_dir):
            QMessageBox.critical(self, "خطا", "مسیر فایل Xray نامعتبر است. لطفاً مسیر را در تب 'بهینه‌ساز فرگمنت' بررسی کنید.")
            return
        
        reply = QMessageBox.question(self, "تایید به‌روزرسانی", 
                                     f"این عملیات آخرین نسخه Xray را دانلود کرده و فایل‌های اجرایی را در پوشه زیر جایگزین می‌کند:\n{xray_dir}\n\nآیا ادامه می‌دهید؟",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.append_log("INFO", "شروع به‌روزرسانی هسته Xray...")
            self.update_xray_action.setEnabled(False)
            
            if sys.platform != "win32":
                QMessageBox.critical(self, "خطا", "به‌روزرسانی خودکار فقط برای ویندوز پشتیبانی می‌شود.")
                self.append_log("ERROR", "به‌روزرسانی خودکار فقط برای ویندوز پشتیبانی می‌شود.")
                self.update_xray_action.setEnabled(True)
                return
            
            temp_zip_path = os.path.join(tempfile.gettempdir(), "xray_update.zip")
            
            self.download_thread = QThread(self)
            worker = DownloadWorker(XRAY_CORE_URL_WINDOWS, temp_zip_path, "Xray-Core-Zip")
            worker.moveToThread(self.download_thread)
            
            worker.finished.connect(lambda success, file_id: self.on_xray_update_finished(success, xray_dir, temp_zip_path))
            worker.finished.connect(self.download_thread.quit)
            worker.finished.connect(worker.deleteLater)
            self.download_thread.finished.connect(self.download_thread.deleteLater)
            self.download_thread.started.connect(worker.run)
            worker.log_message.connect(self.append_log)
            self.download_thread.start()

    def on_xray_update_finished(self, success, xray_dir, zip_path):
        self.update_xray_action.setEnabled(True)
        if not success:
            QMessageBox.critical(self, "خطا", "دانلود هسته Xray ناموفق بود.")
            return

        try:
            import zipfile
            self.append_log("INFO", "در حال استخراج فایل‌های هسته Xray...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.infolist():
                    if member.filename.lower() in ['xray.exe', 'wintun.dll', 'geoip.dat', 'geosite.dat']:
                        zip_ref.extract(member, xray_dir)
                        self.append_log("INFO", f"فایل {member.filename} با موفقیت استخراج شد.")
            QMessageBox.information(self, "موفقیت", "هسته Xray و فایل‌های جانبی با موفقیت به‌روزرسانی شدند.")
        except Exception as e:
            self.append_log("ERROR", f"خطا در استخراج فایل‌ها: {e}")
            QMessageBox.critical(self, "خطا", f"عملیات استخراج فایل‌های Xray ناموفق بود: {e}")
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)

    def fetch_proxies(self):
        url = self.sub_url_edit.text().strip()
        if not url: QMessageBox.warning(self, "خطا", "لطفاً لینک اشتراک را وارد کنید."); return
        self.fetch_button.setEnabled(False); self.fetch_button.setText("در حال دریافت...")
        self.fetch_thread = QThread(self)
        fetch_worker = FetchWorker(url); fetch_worker.moveToThread(self.fetch_thread)
        self.fetch_thread.started.connect(fetch_worker.run)
        fetch_worker.proxies_fetched.connect(self.on_proxies_fetched)
        fetch_worker.fetch_failed.connect(self.on_fetch_failed)
        fetch_worker.finished.connect(self.fetch_thread.quit); fetch_worker.finished.connect(fetch_worker.deleteLater); self.fetch_thread.finished.connect(self.fetch_thread.deleteLater)
        self.fetch_thread.start()

    def on_proxies_fetched(self, proxies):
        self.fetched_proxies = proxies
        self.proxy_combo.clear(); self.proxy_combo.addItems(self.fetched_proxies.keys()); self.proxy_combo.setEnabled(True)
        self.fetch_button.setEnabled(True); self.fetch_button.setText("دریافت پروکسی‌ها")
        self.append_log("INFO", f"{len(proxies)} پروکسی با موفقیت دریافت شد.")

    def on_fetch_failed(self, error_message):
        QMessageBox.critical(self, "خطا", error_message)
        self.fetch_button.setEnabled(True); self.fetch_button.setText("دریافت پروکسی‌ها")

    def start_scan(self):
        if self.is_sub_testing: QMessageBox.warning(self, "توجه", "تست اشتراک در حال اجراست."); return
        if self.thread and self.thread.isRunning(): QMessageBox.warning(self, "توجه", "یک اسکن دیگر در حال اجراست."); return
        settings = self.get_settings();
        if not settings: return
        self.save_best_button.setEnabled(False); self.export_csv_button.setEnabled(False)
        self.best_latency = float('inf'); self.best_params = None; self.best_ping_label.setText("بهترین پینگ: N/A")
        self.append_log("INFO", "="*50); self.append_log("INFO", "شروع فرآیند اسکن هوشمند فرگمنت...")
        config_data = None; original_config_str = ""
        if self.input_tabs.currentIndex() == 0:
            try:
                with open(settings['config_path'], 'r', encoding='utf-8') as f: config_data = json.load(f)
                original_config_str = json.dumps(config_data)
            except Exception as e: QMessageBox.critical(self, "خطا", f"فایل کانفیگ نامعتبر است: {e}"); return
        else:
            if not self.fetched_proxies or self.proxy_combo.currentIndex() < 0: QMessageBox.warning(self, "خطا", "لطفاً ابتدا یک پروکسی از لینک اشتراک انتخاب کنید."); return
            selected_proxy_key = self.proxy_combo.currentText()
            selected_outbound, original_link = self.fetched_proxies[selected_proxy_key]
            if selected_outbound.get("protocol") != "vless": QMessageBox.critical(self, "خطا", "اسکن فرگمنت فقط برای VLESS بهینه شده."); return
            config_data = copy.deepcopy(BASE_CONFIG_TEMPLATE)
            config_data['inbounds'] = [{"port": 0, "protocol": "socks", "tag": "socks-in"}]
            config_data["outbounds"][0] = selected_outbound
            original_config_str = original_link
        self.base_config_for_saving = copy.deepcopy(config_data)
        settings['original_config_str'] = original_config_str
        self.start_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.log_area.clear(); self.results_table.setRowCount(0)
        self.results_table.setSortingEnabled(False)
        self.progress_bar.setValue(0); self.tabs.setCurrentIndex(1)
        self.thread = QThread(self); self.worker = ManagerWorker(settings, config_data); self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.scan_finished); self.worker.finished.connect(self.thread.quit); self.worker.finished.connect(self.worker.deleteLater); self.thread.finished.connect(self.thread.deleteLater)
        self.thread.started.connect(self.worker.run)
        self.worker.log_message.connect(self.append_log); self.worker.progress_updated.connect(self.update_progress); self.worker.result_found.connect(self.add_result_to_table)
        self.thread.start()

    def stop_scan(self):
        if self.worker: self.worker.stop()
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)

    def scan_finished(self):
        self.start_button.setEnabled(True); self.stop_button.setEnabled(False)
        if self.results_table.rowCount() > 0: 
            self.save_best_button.setEnabled(True); self.export_csv_button.setEnabled(True)
            self.results_table.setSortingEnabled(True)
            self.results_table.sortItems(0, Qt.SortOrder.DescendingOrder)
        self.append_log("INFO", "اسکن فرگمنت به پایان رسید.")
        self.thread = None; self.worker = None

    def _start_sub_test(self, configs_to_test, save_mode, num_workers=100):
        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "عملیات نامعتبر", "یک عملیات دیگر در حال اجراست.")
            return

        settings = self.get_settings()
        if not settings: return
        settings['num_workers'] = num_workers
        
        self.sub_test_results_table.setRowCount(0)
        self.sub_test_results_table.setSortingEnabled(False)
        self.update_sub_counters(0, 0, 0)
        self.tabs.setCurrentIndex(2)

        self.thread = QThread(self)
        self.worker = SubscriptionTesterWorkerV2(settings, configs_to_test, save_mode)
        self.worker.moveToThread(self.thread)
        self.worker.log_message.connect(self.append_log)
        self.worker.progress_updated.connect(self.update_sub_counters)
        self.worker.result_found.connect(self.add_sub_test_result)
        self.worker.finished.connect(self.sub_test_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.started.connect(self.worker.run)
        
        self.is_sub_testing = True
        self.sub_fetch_button.setEnabled(False)
        self.sub_stop_button.setEnabled(True)
        
        self.thread.start()

    def stop_sub_test(self):
        if self.worker:
            self.worker.stop()
            self.append_log("INFO", "ارسال فرمان توقف کامل...")
        # Reset UI immediately for better user experience
        self.is_sub_testing = False
        self.sub_fetch_button.setEnabled(True)
        self.sub_stop_button.setEnabled(False)
        if self.thread:
             self.thread.quit()

    def start_sub_test_from_url(self):
        if self.is_sub_testing: return
        
        num_workers = SubTestSettingsDialog.run(self)
        if num_workers is None: return

        profile_name = self.sub_links_combo.currentText()
        if not profile_name:
            QMessageBox.warning(self, "خطا", "لطفاً یک پروفایل اشتراک از لیست انتخاب کنید یا یک پروفایل جدید اضافه کنید.")
            return
        
        url = self.subscription_profiles.get(profile_name)
        if not url:
             QMessageBox.critical(self, "خطا", f"URL برای پروفایل '{profile_name}' یافت نشد.")
             return

        try:
            self.append_log("INFO", f"در حال دریافت کانفیگ از پروفایل: {profile_name}...")
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            raw_content = response.content
            configs = []
            try:
                decoded_content = base64.b64decode(raw_content).decode('utf-8')
                configs = [line.strip() for line in decoded_content.splitlines() if line.strip()]
                self.append_log("INFO", "لینک اشتراک با موفقیت به عنوان Base64 رمزگشایی شد.")
            except (ValueError, UnicodeDecodeError):
                self.append_log("INFO", "لینک اشتراک به عنوان متن ساده پردازش می‌شود.")
                configs = [line.strip() for line in raw_content.decode('utf-8', errors='ignore').splitlines() if line.strip()]

            if not configs:
                self.append_log("ERROR", "هیچ کانفیگی از لینک دریافت نشد."); return
            self._start_sub_test(configs, save_mode='merge', num_workers=num_workers)
        except requests.RequestException as e:
            QMessageBox.critical(self, "خطا", f"خطا در دریافت کانفیگ‌ها: {e}")

    def paste_and_test_configs(self):
        if self.tabs.currentIndex() != 2: return
        if self.is_sub_testing: return

        num_workers = SubTestSettingsDialog.run(self)
        if num_workers is None: return

        try:
            clipboard_text = QApplication.clipboard().text()
            configs = [line.strip() for line in clipboard_text.splitlines() if line.strip()]
            if configs:
                self.append_log("INFO", f"{len(configs)} کانفیگ از کلیپ‌بورد پیست شد.")
                self._start_sub_test(configs, save_mode='test_only', num_workers=num_workers)
        except Exception as e: self.append_log("ERROR", f"خطا در پیست کردن کانفیگ: {e}")

    def sub_test_finished(self, results):
        self.is_sub_testing = False
        self.sub_fetch_button.setEnabled(True)
        self.sub_stop_button.setEnabled(False)
        
        self.sub_test_results_table.setSortingEnabled(True)
        self.sub_test_results_table.sortItems(0, Qt.SortOrder.AscendingOrder)
        self.thread = None; self.worker = None

    def update_sub_counters(self, total, tested, working):
        self.sub_total_label.setText(f"کل: {total}"); self.sub_tested_label.setText(f"تست‌شده: {tested}"); self.sub_working_label.setText(f"سالم: {working}")
        if total > 0: self.progress_bar.setMaximum(total); self.progress_bar.setValue(tested)

    def add_sub_test_result(self, uri, latency, protocol, server, port):
        row_count = self.sub_test_results_table.rowCount()
        self.sub_test_results_table.insertRow(row_count)
        latency_item = QTableWidgetItem(); latency_item.setData(Qt.ItemDataRole.DisplayRole, latency); latency_item.setText(f"{latency:.2f}")
        
        items = [
            latency_item, QTableWidgetItem(protocol), QTableWidgetItem(server),
            QTableWidgetItem(port), QTableWidgetItem(uri)
        ]
        
        for col, item in enumerate(items):
            self.sub_test_results_table.setItem(row_count, col, item)

    def show_sub_table_context_menu(self, pos):
        if not self.sub_test_results_table.selectedItems(): return
        menu = QMenu()
        menu.addAction("کپی لینک").triggered.connect(self.copy_sub_config)
        menu.addAction("حذف").triggered.connect(self.delete_sub_config)
        menu.addAction("نمایش QR Code").triggered.connect(self.show_qr_for_sub_tester)
        menu.exec_(self.sub_test_results_table.mapToGlobal(pos))

    def show_qr_for_sub_tester(self):
        selected_items = self.sub_test_results_table.selectedItems()
        if not selected_items: return
        uri = self.sub_test_results_table.item(selected_items[0].row(), 4).text()
        self.show_qr_code(uri)

    def copy_sub_config(self):
        selected_rows = sorted(list(set(item.row() for item in self.sub_test_results_table.selectedItems())))
        if not selected_rows: return
        uris = [self.sub_test_results_table.item(row, 4).text() for row in selected_rows]
        QApplication.clipboard().setText("\n".join(uris))
        self.append_log("INFO", f"{len(uris)} لینک در کلیپ‌بورد کپی شد.")
        
    def copy_all_working_configs(self):
        row_count = self.sub_test_results_table.rowCount()
        if row_count == 0:
            QMessageBox.information(self, "توجه", "هیچ کانفیگ سالمی در جدول برای کپی وجود ندارد.")
            return
        
        uris = []
        for row in range(row_count):
            item = self.sub_test_results_table.item(row, 4)
            uris.append(item.text())
            
        QApplication.clipboard().setText("\n".join(uris))
        self.append_log("INFO", f"{len(uris)} کانفیگ سالم در کلیپ‌بورد کپی شد.")

    def delete_sub_config(self):
        selected_rows = sorted(list(set(item.row() for item in self.sub_test_results_table.selectedItems())), reverse=True)
        if not selected_rows: return
        uris_to_delete = {self.sub_test_results_table.item(row, 4).text() for row in selected_rows}
        for row in selected_rows: self.sub_test_results_table.removeRow(row)
        try:
            if os.path.exists(BEST_CONFIGS_FILE):
                with open(BEST_CONFIGS_FILE, 'r', encoding='utf-8') as f:
                    current_uris = {line.strip() for line in f if line.strip()}
                remaining_uris = current_uris - uris_to_delete
                with open(BEST_CONFIGS_FILE, 'w', encoding='utf-8') as f:
                    for uri in remaining_uris: f.write(f"{uri}\n")
                self.append_log("INFO", f"{len(uris_to_delete)} کانفیگ از {BEST_CONFIGS_FILE} حذف شد.")
        except Exception as e: self.append_log("ERROR", f"خطا در حذف کانفیگ از فایل: {e}")

    def add_result_to_table(self, result_data, original_config_str):
        row = self.results_table.rowCount(); self.results_table.insertRow(row)
        avg_latency = result_data['avg_latency']
        if avg_latency < self.best_latency:
            self.best_latency = avg_latency
            self.best_params = result_data['params']
            p = self.best_params['packets']; l = self.best_params['length']; i = self.best_params['interval']
            self.best_ping_label.setText(f"بهترین پینگ: {self.best_latency:.2f} ms (P: {p}, L: {l}, I: {i})")
        
        score_item = QTableWidgetItem(); score_item.setData(Qt.DisplayRole, result_data['score']); score_item.setData(Qt.UserRole, original_config_str)
        score_item.setData(Qt.UserRole + 1, json.dumps(self.base_config_for_saving)); score_item.setText(f"{result_data['score']:.2f}")

        latency_item = QTableWidgetItem(f"{avg_latency:.2f}"); speed_item = QTableWidgetItem(f"{result_data['speed']:.2f}" if result_data['speed'] != -1 else "N/A")
        success_item = QTableWidgetItem(f"{result_data['success_rate']*100:.0f}"); jitter_item = QTableWidgetItem(f"{result_data['jitter']:.2f}")
        params = result_data['params']
        items = [score_item, latency_item, speed_item, success_item, jitter_item, QTableWidgetItem(params["packets"]), QTableWidgetItem(params["length"]), QTableWidgetItem(params["interval"])]
        
        color = QColor("#501a1a");
        if result_data['score'] > -10000 and result_data['success_rate'] > 0:
            if avg_latency < 250: color = QColor("#134f13")
            elif avg_latency < 500: color = QColor("#12404a")
            elif avg_latency < 1000: color = QColor("#4f3d13")
        for item in items: item.setBackground(color)
        
        for i, item in enumerate(items): self.results_table.setItem(row, i, item)

    def save_best_config(self):
        if self.results_table.rowCount() == 0 or self.base_config_for_saving is None: QMessageBox.warning(self, "خطا", "هیچ نتیجه موفقی برای ذخیره وجود ندارد."); return
        self.results_table.sortItems(0, Qt.SortOrder.DescendingOrder)
        best_params = {"packets": self.results_table.item(0, 5).text(), "length": self.results_table.item(0, 6).text(), "interval": self.results_table.item(0, 7).text()}
        best_config = copy.deepcopy(self.base_config_for_saving)
        fragment_outbound = next((o for o in best_config.get("outbounds", []) if o.get("tag") == "fragment"), None)
        if fragment_outbound:
            if 'settings' not in fragment_outbound: fragment_outbound['settings'] = {}
            fragment_outbound['settings']['fragment'] = best_params
        save_path, _ = QFileDialog.getSaveFileName(self, "ذخیره بهترین کانفیگ", "config.best.json", "JSON Files (*.json)")
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f: json.dump(best_config, f, indent=2, ensure_ascii=False)
            self.append_log("INFO", f"بهترین کانفیگ در مسیر {save_path} ذخیره شد.")

    def export_results_to_csv(self):
        if self.results_table.rowCount() == 0: QMessageBox.warning(self, "خطا", "هیچ نتیجه‌ای برای خروجی گرفتن وجود ندارد."); return
        path, _ = QFileDialog.getSaveFileName(self, "ذخیره نتایج در فایل CSV", "fragment_results.csv", "CSV Files (*.csv)")
        if path:
            try:
                with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.results_table.horizontalHeaderItem(i).text() for i in range(self.results_table.columnCount())])
                    for row in range(self.results_table.rowCount()): writer.writerow([self.results_table.item(row, col).text() for col in range(self.results_table.columnCount())])
                self.append_log("INFO", f"نتایج با موفقیت در فایل {path} ذخیره شد.")
            except Exception as e: QMessageBox.critical(self, "خطا", f"خطا در ذخیره فایل CSV: {e}")

    def append_log(self, level, message):
        cursor = self.log_area.textCursor(); cursor.movePosition(QTextCursor.MoveOperation.End)
        char_format = cursor.charFormat(); char_format.setForeground(self.LOG_COLORS.get(level, QColor("white"))); cursor.setCharFormat(char_format)
        cursor.insertText(f"[{level}] {message}\n"); self.log_area.ensureCursorVisible()

    def copy_log_to_clipboard(self): QApplication.clipboard().setText(self.log_area.toPlainText()); self.append_log("INFO", "گزارش در کلیپ‌بورد کپی شد.")

    def setup_table_context_menu(self, table, qr_handler): table.setContextMenuPolicy(Qt.CustomContextMenu); table.customContextMenuRequested.connect(lambda pos: self.show_table_context_menu(pos, table, qr_handler))

    def show_table_context_menu(self, pos, table, qr_handler):
        item = table.itemAt(pos)
        if not item: return
        menu = QMenu(); qr_action = QAction("نمایش QR Code", self); qr_action.triggered.connect(lambda: qr_handler(table)); menu.addAction(qr_action)
        menu.exec_(table.mapToGlobal(pos))

    def show_qr_code_for_fragment(self, table):
        selected_items = table.selectedItems()
        if not selected_items: return
        original_config_str = table.item(selected_items[0].row(), 0).data(Qt.UserRole)
        self.show_qr_code(original_config_str)

    def show_qr_code(self, config_str):
        if not (isinstance(config_str, str) and config_str.startswith(('vless://', 'vmess://', 'ss://', 'trojan://'))):
            QMessageBox.information(self, "توجه", "QR Code فقط برای پروتکل‌های پشتیبانی شده قابل نمایش است.")
            return
        try:
            qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
            qr.add_data(config_str)
            qr.make(fit=True)
            pil_img = qr.make_image(fill_color="black", back_color="white")

            buffer = io.BytesIO()
            pil_img.save(buffer, "PNG")
            
            q_pixmap = QPixmap()
            q_pixmap.loadFromData(buffer.getvalue())

            dlg = QMessageBox(self)
            dlg.setWindowTitle("QR Code")
            dlg.setIconPixmap(q_pixmap.scaled(350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            dlg.exec()
        except Exception as e:
            self.append_log("ERROR", f"خطا در تولید QR Code: {e}")
            QMessageBox.critical(self, "خطا", f"خطا در تولید QR Code: {e}")

    def update_progress(self, current, total): self.progress_bar.setMaximum(total); self.progress_bar.setValue(current)

    def get_settings(self):
        xray_path = self.xray_path_edit.text()
        if not os.path.exists(xray_path): QMessageBox.critical(self, "خطا", "فایل اجرایی Xray یافت نشد!"); return None
        config_path = self.config_path_edit.text() if self.input_tabs.currentIndex() == 0 else ""
        if self.input_tabs.currentIndex() == 0 and not os.path.exists(config_path): QMessageBox.critical(self, "خطا", "فایل کانفیگ یافت نشد!"); return None
        
        selected_profile = self.url_combo.currentText()
        urls_str = ""
        if selected_profile == "سفارشی":
            urls_str = self.custom_url_edit.text()
        else:
            urls_str = self.URL_PROFILES.get(selected_profile, "")

        urls = [url.strip() for url in urls_str.split(',') if url.strip()]
        if not urls:
            QMessageBox.critical(self, "خطا", "حداقل یک URL برای تست مورد نیاز است.")
            return None
            
        return {
            "xray_path": xray_path,
            "config_path": config_path,
            "urls": urls,
            "timeout": self.timeout_spin.value(),
            "num_workers": self.cpu_spin.value(),
            "packets_list": [line for line in self.packets_edit.toPlainText().split('\n') if line.strip()],
            "length_list": [line for line in self.length_edit.toPlainText().split('\n') if line.strip()],
            "interval_list": [line for line in self.interval_edit.toPlainText().split('\n') if line.strip()]
        }

    def browse_path(self, line_edit, caption, file_filter="All Files (*)"):
        path, _ = QFileDialog.getOpenFileName(self, caption, "", file_filter)
        if path: line_edit.setText(path)

    def save_settings(self):
        settings = {
            "xray_path": self.xray_path_edit.text(), "config_path": self.config_path_edit.text(), 
            "sub_url": self.sub_url_edit.text(), "url_profile": self.url_combo.currentText(),
            "custom_url": self.custom_url_edit.text(),
            "timeout": self.timeout_spin.value(), "cpu_cores": self.cpu_spin.value(), 
            "profile": self.profile_combo.currentText(), 
            "custom_packets": self.packets_edit.toPlainText(), 
            "custom_length": self.length_edit.toPlainText(), 
            "custom_interval": self.interval_edit.toPlainText(),
            "geometry": self.saveGeometry().toHex().data().decode(),
            "active_tab": self.tabs.currentIndex(),
            "results_cols": [self.results_table.columnWidth(i) for i in range(self.results_table.columnCount())],
            "sub_cols": [self.sub_test_results_table.columnWidth(i) for i in range(self.sub_test_results_table.columnCount())],
            "subscription_profiles": self.subscription_profiles,
            "selected_subscription_index": self.sub_links_combo.currentIndex()
        }
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(settings, f, indent=2, ensure_ascii=False)

    def load_settings(self):
        if not os.path.exists(SETTINGS_FILE):
            self.subscription_profiles = self.DEFAULT_SUBSCRIPTIONS.copy()
            self.sub_links_combo.addItems(self.subscription_profiles.keys())
            self.profile_combo.setCurrentText("free")
            self.apply_test_profile("free")
            return
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f: settings = json.load(f)
            self.xray_path_edit.setText(settings.get("xray_path", os.path.abspath(XRAY_EXECUTABLE)))
            self.config_path_edit.setText(settings.get("config_path", os.path.abspath("config.json")))
            self.sub_url_edit.setText(settings.get("sub_url", ""))
            
            self.url_combo.setCurrentText(settings.get("url_profile", "YouTube")) 
            self.custom_url_edit.setText(settings.get("custom_url", ""))
            
            self.timeout_spin.setValue(settings.get("timeout", 2))
            self.cpu_spin.setValue(settings.get("cpu_cores", 6))
            
            profile = settings.get("profile", "free"); self.profile_combo.setCurrentText(profile)
            if profile == "سفارشی":
                self.packets_edit.setPlainText(settings.get("custom_packets", ""))
                self.length_edit.setPlainText(settings.get("custom_length", ""))
                self.interval_edit.setPlainText(settings.get("custom_interval", ""))
            
            if "geometry" in settings: self.restoreGeometry(QByteArray.fromHex(settings["geometry"].encode()))
            if "active_tab" in settings: self.tabs.setCurrentIndex(settings["active_tab"])
            if "results_cols" in settings:
                for i, width in enumerate(settings.get("results_cols", [])): self.results_table.setColumnWidth(i, width)
            if "sub_cols" in settings:
                for i, width in enumerate(settings.get("sub_cols", [])): self.sub_test_results_table.setColumnWidth(i, width)
            
            profiles = settings.get("subscription_profiles", {})
            if not profiles:
                self.subscription_profiles = self.DEFAULT_SUBSCRIPTIONS.copy()
            else:
                self.subscription_profiles = profiles
                
            self.sub_links_combo.addItems(self.subscription_profiles.keys())
            index = settings.get("selected_subscription_index", -1)
            if index != -1 and index < self.sub_links_combo.count():
                self.sub_links_combo.setCurrentIndex(index)

        except (json.JSONDecodeError, KeyError) as e: self.append_log("ERROR", f"فایل settings.json نامعتبر یا ناقص است: {e}")

    def add_subscription_profile(self):
        name = self.new_sub_name_edit.text().strip()
        link = self.new_sub_link_edit.text().strip()
        if name and link:
            if name not in self.subscription_profiles:
                self.subscription_profiles[name] = link
                self.sub_links_combo.addItem(name)
                self.sub_links_combo.setCurrentText(name)
                self.new_sub_name_edit.clear()
                self.new_sub_link_edit.clear()
                self.save_settings()
                self.append_log("INFO", "پروفایل اشتراک جدید اضافه شد.")
            else:
                QMessageBox.information(self, "توجه", "پروفیلی با این نام قبلاً اضافه شده است.")
        else:
            QMessageBox.warning(self, "خطا", "لطفاً نام و لینک پروفایل را وارد کنید.")
            
    def remove_subscription_profile(self):
        index = self.sub_links_combo.currentIndex()
        if index != -1:
            name = self.sub_links_combo.currentText()
            if name in self.subscription_profiles:
                del self.subscription_profiles[name]
                self.sub_links_combo.removeItem(index)
                self.save_settings()
                self.append_log("INFO", f"پروفایل حذف شد: {name}")
        else:
            QMessageBox.warning(self, "خطا", "لطفاً ابتدا یک پروفایل را از لیست انتخاب کنید.")

    def closeEvent(self, event):
        self.save_settings()
        if self.thread and self.thread.isRunning():
            reply = QMessageBox.question(self, "خروج از برنامه", 
                                         "یک عملیات در حال اجراست. آیا می‌خواهید آن را متوقف کرده و خارج شوید؟",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                if self.worker:
                    self.worker.stop()
                self.thread.quit()
                if not self.thread.wait(3000):
                    self.thread.terminate()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
            
    def __del__(self):
        kill_xray_processes()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(kill_xray_processes)
    window.show()
    sys.exit(app.exec())