import cv2
import numpy as np
import sqlite3
import json
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.fatigue_detection import FatigueDetector

class FaceRecognition:
    def __init__(self, db_path='mrsoft.db'):
        self.db_path = db_path
        self.fatigue_detector = FatigueDetector()
        # 初始化时不创建数据库连接，而是在每个方法中创建
        self.create_tables()
    
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
        """注册用户"""
        conn = self.get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
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
    
    def register_face(self, user_id, name, face_img):
        """注册人脸"""
        try:
            # 提取人脸特征
            face_feat = self.fatigue_detector.get_face_feat(face_img)
            if face_feat is not None:
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
                print("特征提取失败")
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
                        features = np.array(json.loads(features_json))
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
            # 提取当前人脸特征
            current_feat = self.fatigue_detector.get_face_feat(face_img)
            if current_feat is None:
                print("特征提取失败")
                return None
            
            # 获取数据库中的所有人脸特征
            face_data = self.get_all_face_features()
            if not face_data:
                print("数据库中没有人脸数据")
                return None
            
            # 计算相似度
            min_dist = float('inf')
            best_match = None
            
            print(f"数据库中有 {len(face_data)} 个人脸数据")
            for face_id, name, db_feat in face_data:
                print(f"比较人脸: {name}")
                try:
                    dist = np.linalg.norm(current_feat - db_feat)
                    print(f"距离: {dist}")
                    if dist < min_dist:
                        min_dist = dist
                        best_match = name
                except Exception as e:
                    print(f"计算距离失败: {e}")
                    continue
            
            # 设置阈值，只有当距离小于阈值时才认为匹配成功
            print(f"最小距离: {min_dist}")
            # 调整阈值为 0.8，提高识别成功率
            if min_dist < 0.8:
                print(f"识别成功: {best_match}")
                return best_match
            else:
                print("识别失败: 距离超过阈值")
                return None
        except Exception as e:
            print(f"人脸识别失败: {e}")
            return None
    
    def close_db(self):
        """关闭数据库连接"""
        # 由于每个方法都创建了自己的连接，这里不需要做任何事情
        print("数据库连接管理完成")