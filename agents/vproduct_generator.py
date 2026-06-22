import os
import re
import subprocess
import sys
import json


class VideoProductGenerator:
    def __init__(self, ideas) -> None:
        self.ideas = ideas

    def __call__(self):
        current_dir = os.getcwd()
        video_agent_dir = os.path.join(current_dir, 'VideoAgent')

        if not os.path.exists(video_agent_dir):
            print(f"错误: VideoAgent目录不存在: {video_agent_dir}")

        cmd = [sys.executable, 'main.py', '--ideas', json.dumps(self.ideas)]

        try:
            print(f"🚀 启动VideoAgent: {' '.join(cmd)}")
            print(f"📁 工作目录: {video_agent_dir}")
            print("-" * 50)

            # 不重定向输入输出，允许交互式输入
            process = subprocess.Popen(
                cmd,
                cwd=video_agent_dir
            )

            # 等待子进程完成
            return_code = process.wait()

            print("-" * 50)
            if return_code == 0:
                print("✅ VideoAgent 执行成功")
            else:
                print(f"❌ VideoAgent 执行失败，退出代码: {return_code}")

            return return_code == 0

        except KeyboardInterrupt:
            print("\n⚠️  用户中断执行")
            process.terminate()
            process.wait()
            return False
        except Exception as e:
            print(f"❌ 执行出错: {e}")
            return False
