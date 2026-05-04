import cv2
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import sys

import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.fatigue_detection import FatigueDetector

_PBKDF2_ITERS = 200_000
_PWD_PREFIX = "pbkdf2_sha256$"


def _hash_password(plain: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, _PBKDF2_ITERS)
    return _PWD_PREFIX + salt.hex() + "$" + dk.hex()


def _verify_password(plain: str, stored: str) -> bool:
    if stored.startswith(_PWD_PREFIX):
        try:
            rest = stored[len(_PWD_PREFIX) :]
            salt_hex, want_hex = rest.split("$", 1)
            salt = bytes.fromhex(salt_hex)
            dk = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, _PBKDF2_ITERS)
            return hmac.compare_digest(dk.hex(), want_hex)
        except Exception:
            return False
    # 兼容历史明文密码（登录成功后会自动升级为哈希）
    try:
        return hmac.compare_digest(stored.encode("utf-8"), plain.encode("utf-8"))
    except Exception:
        return False

class FaceRecognition:
    def __init__(self, db_path="mrsoft.db", fatigue_detector=None, eager_load_detector=True):
        self.db_path = db_path
        if fatigue_detector is not None:
            self.fatigue_detector = fatigue_detector
        elif eager_load_detector:
            self.fatigue_detector = FatigueDetector()
        else:
            self.fatigue_detector = None
        self.create_tables()
        # 归一化后的欧氏距离阈值（越接近 0 越相似；1.2 比原 0.8 更宽松，适配不同裁剪/光照）
        self.match_thresh = 1.2
        self._warned_embedder = False

    def embedder_status(self):
        """
        返回 (ok, reason)。用于解释为什么无法提取 embedding。
        """
        fd = self.fatigue_detector
        if fd is None:
            return False, "FatigueDetector 未初始化"
        err = getattr(fd, "last_model_error", None)
        if err:
            # FaceLandmarker 失败不等于 FaceNet 失败；这里只作为补充信息
            pass
        try:
            import torch  # noqa: F401
        except Exception:
            return False, "未安装 PyTorch：当前 Python 环境 import torch 失败（人脸特征提取不可用）"

        if getattr(fd, "face_net", None) is None or getattr(fd, "device", None) is None:
            return False, "FaceNet 未加载：请检查 resources/weights/facenet_best_server.pt 是否存在，以及依赖是否齐全"

        return True, ""

    @staticmethod
    def _flatten_embedding(feat):
        """将网络输出特征整理成 1D 向量，避免 (1,128) vs (128,) 距离计算异常。"""
        if feat is None:
            return None
        arr = np.asarray(feat, dtype=np.float32).reshape(-1)
        return arr

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return v
        return (v / n).astype(np.float32)
    
    def get_db_connection(self):
        """获取数据库连接，每个线程都创建自己的连接"""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            print(f"数据库连接错误: {e}")
            return None
    
    def create_tables(self):
        """创建数据库表"""
        conn = self.get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                # 创建用户表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL
                    )
                ''')
                
                # 创建人脸特征表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        features TEXT NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                ''')
                conn.commit()
                print("数据库表创建成功")
            except sqlite3.Error as e:
                print(f"数据库表创建错误: {e}")
            finally:
                conn.close()
    
    def register_user(self, username, password):
        """注册用户（密码以 PBKDF2 存储）"""
        conn = self.get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                ph = _hash_password(password)
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, ph))
                conn.commit()
                user_id = cursor.lastrowid
                print(f"用户注册成功，ID: {user_id}")
                return user_id
            except sqlite3.IntegrityError:
                print("用户名已存在")
                return None
            except Exception as e:
                print(f"注册失败: {e}")
                return None
            finally:
                conn.close()
        return None

    def verify_user_login(self, username, password):
        """
        验证账号密码。成功返回 user_id；失败返回 None。
        若库中为历史明文密码且校验通过，会自动写回哈希。
        """
        conn = self.get_db_connection()
        if not conn:
            return None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, password FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            if not row:
                return None
            user_id, stored = int(row[0]), row[1]
            if not _verify_password(password, stored):
                return None
            if not str(stored).startswith(_PWD_PREFIX):
                cursor.execute(
                    "UPDATE users SET password = ? WHERE id = ?",
                    (_hash_password(password), user_id),
                )
                conn.commit()
            return user_id
        except Exception as e:
            print(f"登录校验失败: {e}")
            return None
        finally:
            conn.close()

    def register_face(self, user_id, name, face_img):
        """注册人脸"""
        try:
            ok, reason = self.embedder_status()
            if not ok:
                print(f"人脸注册不可用: {reason}")
                return False

            # 提取人脸特征
            face_feat = self.fatigue_detector.get_face_feat(face_img)
            face_feat = self._flatten_embedding(face_feat)
            if face_feat is not None:
                face_feat = self._l2_normalize(face_feat)
                # 将特征转换为JSON格式
                features_json = json.dumps(face_feat.tolist())
                # 插入数据库
                conn = self.get_db_connection()
                if conn:
                    try:
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO faces (user_id, name, features) VALUES (?, ?, ?)", (user_id, name, features_json))
                        conn.commit()
                        print("人脸注册成功")
                        return True
                    except Exception as e:
                        print(f"人脸注册失败: {e}")
                        return False
                    finally:
                        conn.close()
                else:
                    print("数据库连接失败")
                    return False
            else:
                print("特征提取失败：人脸图像过小/通道不对，或模型前向失败")
                return False
        except Exception as e:
            print(f"人脸注册失败: {e}")
            return False
    
    def get_all_face_features(self):
        """获取所有人脸特征"""
        try:
            import numpy as np
            conn = self.get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id, name, features FROM faces")
                    rows = cursor.fetchall()
                    face_data = []
                    for row in rows:
                        face_id, name, features_json = row
                        features = self._flatten_embedding(json.loads(features_json))
                        face_data.append((face_id, name, features))
                    return face_data
                except Exception as e:
                    print(f"获取人脸特征失败: {e}")
                    return []
                finally:
                    conn.close()
            return []
        except Exception as e:
            print(f"获取人脸特征失败: {e}")
            return []
    
    def recognize_face(self, face_img):
        """识别人脸"""
        try:
            import numpy as np
            ok, reason = self.embedder_status()
            if not ok:
                if not self._warned_embedder:
                    print(f"人脸识别不可用: {reason}")
                    self._warned_embedder = True
                return None

            # 提取当前人脸特征
            current_feat = self.fatigue_detector.get_face_feat(face_img)
            current_feat = self._flatten_embedding(current_feat)
            if current_feat is None:
                return None
            current_feat = self._l2_normalize(current_feat)
            
            # 获取数据库中的所有人脸特征
            face_data = self.get_all_face_features()
            if not face_data:
                if os.environ.get("TIRED_FACE_RECO_DEBUG") == "1":
                    print("数据库中没有人脸数据")
                return None
            
            # 计算相似度
            min_dist = float('inf')
            best_match = None
            _dbg = os.environ.get("TIRED_FACE_RECO_DEBUG") == "1"
            if _dbg:
                print(f"数据库中有 {len(face_data)} 个人脸数据")
            for face_id, name, db_feat in face_data:
                if _dbg:
                    print(f"比较人脸: {name}")
                try:
                    db_vec = self._l2_normalize(self._flatten_embedding(db_feat))
                    dist = float(np.linalg.norm(current_feat - db_vec))
                    if _dbg:
                        print(f"距离: {dist}")
                    if dist < min_dist:
                        min_dist = dist
                        best_match = name
                except Exception as e:
                    if _dbg:
                        print(f"计算距离失败: {e}")
                    continue
            
            if _dbg:
                print(f"最小距离: {min_dist}")
            if min_dist < self.match_thresh:
                if _dbg:
                    print(f"识别成功: {best_match}")
                return best_match
            if _dbg:
                print("识别失败: 距离超过阈值")
            return None
        except Exception as e:
            print(f"人脸识别失败: {e}")
            return None
    
    def close_db(self):
        """关闭数据库连接"""
        # 由于每个方法都创建了自己的连接，这里不需要做任何事情
        print("数据库连接管理完成")