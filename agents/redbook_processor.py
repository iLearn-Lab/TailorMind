import os
import sys
import json
import glob
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add agents directory to path if running from project root
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from text_analyst import TextAnalyst
from image_analyst import ImageAnalyst
from video_analyst import VideoAnalyst
from user_profile_generator import UserProfileGenerator
from profile2idea_it import Profile2Idea
from itproduct_generator import ITProductGenerator, AIRefusalError
from redbook_english import RedBookEnglishConverter


class RedBookProcessor:
    def __init__(self, data_root="download/redbook", output_base="generated_redbook_it"):
        self.data_root = data_root
        self.output_base = output_base
        os.makedirs(output_base, exist_ok=True)

    def setup_environment(self):
        """设置环境变量"""
        os.environ.update({
            "CHAT_API_KEY": "sk-dVaSXmTEMBh0Gygx49ResSvaONvErml5QV8McBAGkbPmX2mG",
            "CHAT_BASE_URL": "https://yunwu.ai/v1",
            "CHAT_MODEL": "gpt-4o",

            "IMAGE_API_KEY": "sk-dVaSXmTEMBh0Gygx49ResSvaONvErml5QV8McBAGkbPmX2mG",
            "IMAGE_BASE_URL": "https://yunwu.ai/v1",
            "IMAGE_MODEL": "gpt-4-vision-preview",

            "VIDEO_API_KEY": "sk-dVaSXmTEMBh0Gygx49ResSvaONvErml5QV8McBAGkbPmX2mG",
            "VIDEO_BASE_URL": "https://yunwu.ai/v1",
            "VIDEO_MODEL": "gpt-4o",

            "GENERATE_API_KEY": "sk-W7qxtvbQUxwIo9PLlSGh89cUKKuTTo1oUXmqpGoYIhqQULjI",
            "GENERATE_BASE_URL": "https://yunwu.ai/v1beta",
            "GENERATE_MODEL": "gemini-2.5-flash-image-preview",

            "IDEA_API_KEY": "sk-dVaSXmTEMBh0Gygx49ResSvaONvErml5QV8McBAGkbPmX2mG",
            "IDEA_BASE_URL": "https://yunwu.ai/v1",
            "IDEA_MODEL": "gpt-4o",
        })

    def get_available_users(self) -> List[str]:
        users = []
        if os.path.exists(self.data_root):
            for name in os.listdir(self.data_root):
                user_path = os.path.join(self.data_root, name)
                if os.path.isdir(os.path.join(user_path, "historical")):
                    users.append(name)
        return sorted(users)


    def load_user_data(self, user_id: str) -> Optional[Dict]:
        user_path = os.path.join(self.data_root, user_id)
        historical_path = os.path.join(user_path, "historical")

        if not os.path.isdir(historical_path):
            print(f"historical 目录不存在: {historical_path}")
            return None

        items = []
        for item_dir in os.listdir(historical_path):
            item_path = os.path.join(historical_path, item_dir)
            if os.path.isdir(item_path):
                item_data = self._parse_item(historical_path, item_dir)
                if item_data:
                    items.append(item_data)

        return {
            "user_id": user_id,
            "total_items": len(items),
            "items": items
        }


    def _parse_item(self, user_path: str, item_dir: str) -> Optional[Dict]:
        """解析单个收藏项"""
        item_path = os.path.join(user_path, item_dir)
        base_name = '_'.join(item_dir.split('_')[1:])  # 去掉itemX_前缀

        # 查找文本文件
        text_files = glob.glob(os.path.join(item_path, "*.txt"))
        text_content = ""
        if text_files:
            try:
                with open(text_files[0], 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
            except:
                print(f"无法读取文本文件: {text_files[0]}")

        # 查找图片文件 - 使用更精确的匹配
        image_files = []

        # 优先匹配带_img编号的图片
        numbered_images = glob.glob(os.path.join(item_path, "*_img[0-9]*.jpg")) + \
                        glob.glob(os.path.join(item_path, "*_img[0-9]*.png"))

        # 匹配其他图片文件，但排除缩略图和小图
        other_images = glob.glob(os.path.join(item_path, "*.jpg")) + \
                    glob.glob(os.path.join(item_path, "*.png"))

        # 过滤掉可能的重复和缩略图
        all_images = set()  # 使用set去重

        for img_path in numbered_images + other_images:
            # 排除常见的缩略图命名
            if any(thumb in os.path.basename(img_path).lower() for thumb in ['thumb', 'small', 'mini', '_s.']):
                continue

            # 排除_img0（通常是封面缩略图）
            if '_img0.' in img_path.lower():
                continue

            # 使用绝对路径避免重复
            abs_path = os.path.abspath(img_path)
            all_images.add(abs_path)

        image_files = list(all_images)

        # 查找视频文件
        video_files = glob.glob(os.path.join(item_path, "*.mp4")) + \
                    glob.glob(os.path.join(item_path, "*.mov"))

        if not text_content and not image_files and not video_files:
            return None

        return {
            "item_id": item_dir,
            "item_name": base_name,
            "text_content": text_content,
            "image_files": image_files,
            "video_files": video_files,
            "item_path": item_path
        }

    def process_user(self, user_id: str, max_workers: int = 4, user_index: Optional[int] = None, generate_english: bool = True) -> Optional[Dict]:
        """处理单个用户数据并生成个性化帖子"""
        print(f"\n{'='*50}")
        print(f"开始处理用户 [{user_index}] {user_id}" if user_index is not None else f"开始处理用户 {user_id}")
        print(f"{'='*50}")

        # 加载用户数据 ———— 因为已生成profile，暂且关闭
        # user_data = self.load_user_data(user_id)
        # if not user_data or user_data["total_items"] == 0:
        #     print(f"用户 {user_id} 无有效数据，跳过处理")
        #     return None

        # print(f"找到 {user_data['total_items']} 个收藏项")

        # 创建输出目录（带序号）
        if user_index is not None:
            output_dir_name = f"{user_index}_{user_id}"
        else:
            output_dir_name = user_id
        user_output_dir = os.path.join(self.output_base, output_dir_name)
        os.makedirs(user_output_dir, exist_ok=True)

        # 断点续传检查：如果final_results.json已存在，跳过处理
        final_results_path = os.path.join(user_output_dir, "final_results.json")
        if os.path.exists(final_results_path):
            try:
                # 验证文件是否有效（包含必要字段）
                with open(final_results_path, 'r', encoding='utf-8') as f:
                    existing_result = json.load(f)
                    if existing_result.get("personalized_post") or existing_result.get("discussion_post"):
                        print(f"✅ 检测到已存在的final_results.json，跳过处理（断点续传）")
                        print(f"   📄 文件路径: {final_results_path}")
                        return existing_result
            except (json.JSONDecodeError, Exception) as e:
                print(f"⚠️  检测到final_results.json但解析失败: {e}，将重新生成")
                # 继续处理，覆盖旧文件

        temp_files = []  # 记录所有临时文件路径

        
        try:
            # ===== analysis =====
            # # 分析文本内容
            # text_files = []
            # for item in user_data["items"]:
            #     if item["text_content"]:
            #         # 将文本内容保存为临时文件供分析
            #         text_file = os.path.join(user_output_dir, f"temp_{item['item_id']}.txt")
            #         with open(text_file, "w", encoding="utf-8") as f:
            #             f.write(item["text_content"])
            #         text_files.append(text_file)
            #         temp_files.append(text_file)  # 记录临时文件

            # if text_files:
            #     print("分析文本内容...")
            #     text_analyst = TextAnalyst(max_workers=max_workers)
            #     text_analyst(text_files, user_output_dir)
            # else:
            #     print("无文本内容可分析")

            # # 分析图片内容
            # image_files = []
            # for item in user_data["items"]:
            #     image_files.extend(item["image_files"])

            # if image_files:
            #     print(f"分析 {len(image_files)} 张图片...")
            #     image_analyst = ImageAnalyst(max_workers=max_workers)
            #     image_analyst(image_files, user_output_dir)
            # else:
            #     print("无图片可分析")

            # 分析视频内容 --------------------- 暂且关闭
            # video_files = []
            # for item in user_data["items"]:
            #     video_files.extend(item["video_files"])

            # if video_files:
            #     print(f"分析 {len(video_files)} 个视频...")
            #     video_analyst = VideoAnalyst(max_workers=max_workers)
            #     video_analyst(video_files, user_output_dir)
            # else:
            #     print("无视频可分析")


            # # ===== analysis -> profile =====
            # print("生成用户画像...")
            # analysis_data = self._combine_analysis_results(user_output_dir)

            # if not analysis_data.strip():
            #     print("无分析数据，无法生成用户画像")
            #     return None

            # profile_generator = UserProfileGenerator()
            # user_profile = profile_generator(analysis_data)
    

            # ===== 直接使用已有的 user_profile.txt =====
            print("读取已有的用户画像...")
            
            # 从 download/redbook/{user_id}/user_profile.txt 读取
            source_profile_path = os.path.join(self.data_root, user_id, "user_profile.txt")
            
            if not os.path.exists(source_profile_path):
                print(f"❌ 未找到用户画像文件: {source_profile_path}")
                print("   尝试生成新的用户画像...")
                
                # 如果不存在，回退到原来的生成逻辑
                analysis_data = self._combine_analysis_results(user_output_dir)
                if not analysis_data.strip():
                    print("无分析数据，无法生成用户画像")
                    return None
                    
                profile_generator = UserProfileGenerator()
                user_profile_text = profile_generator(analysis_data)
                
                # Extract first preference only
                first_preference = self._extract_first_preference(user_profile_text)
                
                # 转换为 JSON 格式
                user_profile = {
                    "user_id": user_id,
                    "profile_text": first_preference,
                    "source": "generated",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # 读取已有的 user_profile.txt
                with open(source_profile_path, "r", encoding="utf-8") as f:
                    user_profile_text = f.read().strip()
                
                print(f"✅ 成功读取用户画像 (长度: {len(user_profile_text)} 字符)")
                
                # Extract first preference only
                first_preference = self._extract_first_preference(user_profile_text)
                
                # 转换为 JSON 格式
                user_profile = {
                    "user_id": user_id,
                    "profile_text": first_preference,
                    "source": source_profile_path,
                    "timestamp": datetime.now().isoformat()
                }

            # 保存用户画像到 generated_it/{user_id}/profile.json
            profile_path = os.path.join(user_output_dir, "profile.json")
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(user_profile, f, ensure_ascii=False, indent=2)
            
            print(f"   画像已保存到: {profile_path}")

            # ===== profile -> idea（只使用 Text-Image Content）=====
            print("根据用户画像生成内容创意（Text-Image Content）...")

            profile2idea = Profile2Idea()

            # 准备传给 profile2idea 的内容
            if isinstance(user_profile, dict) and "profile_text" in user_profile:
                # 如果是从 user_profile.txt 读取的，使用 profile_text
                profile_content = user_profile["profile_text"]
            elif isinstance(user_profile, dict):
                # 如果是字典但没有 profile_text，转为 JSON 字符串
                profile_content = json.dumps(user_profile, ensure_ascii=False, indent=2)
            else:
                # 如果已经是字符串，直接使用
                profile_content = user_profile

            raw_ideas = profile2idea(
                user_profile=profile_content,
                user_dir=user_output_dir
            )

            ideas = json.loads(raw_ideas)

            # 只保留 Text-Image Content
            ideas = [
                idea for idea in ideas
                if idea.get("main_type") == "Text-Image Content"
            ]

            if not ideas:
                raise ValueError("❌ 未生成任何 Text-Image Content 类型的 ideas")

            # 只使用第一个 idea
            if len(ideas) > 1:
                print(f"✅ 生成 {len(ideas)} 个 Text-Image Content ideas，只使用第一个")
                ideas = [ideas[0]]
            else:
                print(f"✅ 生成 {len(ideas)} 个 Text-Image Content idea")

            # ===== idea -> post =====
            # 生成个性化帖子
            print("生成个性化帖子...")
            content_generator = ITProductGenerator(ideas)
            
            try:
                # 正确的参数顺序：user_profile, output_dir, user_profile_path
                personalized_post = content_generator(user_profile, user_output_dir, profile_path)
            except AIRefusalError as e:
                print(f"\n{'='*50}")
                print(f"⚠️  AI连续拒绝生成内容，跳过用户 {user_id}")
                print(f"{'='*50}\n")
                return None

            # Save intermediate results (without english_post) for English converter to read
            intermediate_result = {
                "user_id": user_id,
                "user_profile": user_profile,
                "personalized_post": personalized_post
            }
            
            with open(os.path.join(user_output_dir, "final_results.json"), "w", encoding="utf-8") as f:
                json.dump(intermediate_result, f, ensure_ascii=False, indent=2)

            # ===== Auto-generate English version =====
            english_post = None
            if generate_english:
                try:
                    print("\n🌍 生成英文版帖子（国际社交媒体风格）...")
                    
                    # 创建英文转换器
                    english_converter = RedBookEnglishConverter(generated_dir=self.output_base)
                    
                    # 转换为英文
                    user_dir_name = output_dir_name  # 使用之前定义的目录名
                    english_post = english_converter.convert_post(user_dir_name)
                    
                    if english_post:
                        print(f"✅ 英文版生成成功！")
                        print(f"📄 英文HTML: {english_post['html_path']}")
                    else:
                        print(f"⚠️ 英文版生成失败，但中文版已成功生成")
                
                except Exception as e:
                    print(f"⚠️ 生成英文版时出错: {e}")
                    print(f"中文版已成功生成，可稍后手动转换")
                    import traceback
                    traceback.print_exc()

            # Save final results (update with english_post)
            final_result = {
                "user_id": user_id,
                "user_profile": user_profile,
                "personalized_post": personalized_post,
                "english_post": english_post
            }

            with open(os.path.join(user_output_dir, "final_results.json"), "w", encoding="utf-8") as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)

            img_count = len(personalized_post.get('images', []))
            link_count = len(personalized_post.get('links', []))
            
            print(f"\n✅ 用户 {user_id} 处理完成！")
            print(f"📝 帖子文案: {len(personalized_post['text'])} 字")
            print(f"🖼️  生成图片: {img_count} 张")
            print(f"🔗 相关链接: {link_count} 个")
            print(f"📄 HTML文件: {personalized_post['html_post']}")
            if english_post:
                print(f"🌍 英文HTML: {english_post['html_path']}")
            
            # 清理该用户的embedding缓存
            try:
                content_generator.rag_helper.clear_user_cache("redbook", user_id)
            except Exception as e:
                print(f"⚠️  清理embedding缓存失败: {e}")

            return final_result

        finally:
            # 清理临时文件
            self._cleanup_temp_files(temp_files)

    def _cleanup_temp_files(self, temp_files):
        """清理临时文件"""
        cleaned_count = 0
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    cleaned_count += 1
            except Exception as e:
                print(f"删除临时文件失败 {temp_file}: {e}")

        if cleaned_count > 0:
            print(f"🧹 已清理 {cleaned_count} 个临时文件")

    def _extract_first_preference(self, user_profile_text: str) -> str:
        """Extract the first preference from user profile text.
        
        The user profile format may vary, but always contains:
        "Ordering by user preference level, from highest to lowest:"
        
        Returns the first preference section with the marker line included.
        """
        # Find the marker line (may have ** markdown formatting)
        lines = user_profile_text.split('\n')
        marker_idx = -1
        marker_line = None
        
        for i, line in enumerate(lines):
            # Check if line contains the ordering marker (with or without markdown)
            if "Ordering by user preference level, from highest to lowest" in line:
                marker_idx = i
                marker_line = line
                break
        
        if marker_idx == -1:
            # If marker not found, return original text
            print("⚠️  未找到 preference 标记，使用完整用户画像")
            return user_profile_text
        
        # Extract first preference
        # Look for the start of the first preference (usually "## 1." or "1. Preference 1:" or similar)
        first_pref_start = -1
        for i in range(marker_idx + 1, len(lines)):
            line = lines[i].strip()
            # Skip empty lines
            if not line:
                continue
            # Check if this is the start of first preference
            # Pattern: "## 1.", "1. Preference 1:", "1. ", etc.
            if (line.startswith("## 1.") or 
                line.startswith("**## 1.") or
                (line.startswith("1.") and ("Preference 1" in line or len(line) > 3)) or
                line.startswith("1. Preference 1")):
                first_pref_start = i
                break
        
        if first_pref_start == -1:
            # If can't find first preference start, return from marker onwards
            first_pref_start = marker_idx + 1
        
        # Find the end of first preference (start of second preference or end of text)
        first_pref_end = len(lines)
        for i in range(first_pref_start + 1, len(lines)):
            line = lines[i].strip()
            # Check if this is the start of second preference
            if (line.startswith("## 2.") or 
                line.startswith("**## 2.") or
                (line.startswith("2.") and ("Preference 2" in line or len(line) > 3)) or
                line.startswith("2. Preference 2")):
                first_pref_end = i
                break
        
        # Extract the first preference section, including the marker line
        first_preference_lines = [marker_line] + lines[first_pref_start:first_pref_end]
        first_preference = '\n'.join(first_preference_lines).strip()
        
        if not first_preference:
            print("⚠️  未能提取到第一个 preference，使用完整用户画像")
            return user_profile_text
        
        print(f"✅ 已提取 Preference 1 (长度: {len(first_preference)} 字符)")
        return first_preference

    def _combine_analysis_results(self, user_output_dir: str) -> str:
        """合并分析结果"""
        analysis_path = os.path.join(user_output_dir, "analysis.json")
        if not os.path.exists(analysis_path):
            return ""

        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)

        combined_analysis = ""
        for media_type, contents in analysis_data.items():
            combined_analysis += f"{media_type.upper()}分析结果:\n"
            for idx, content in contents.items():
                combined_analysis += f"项目{idx}: {content}\n\n"

        return combined_analysis

    def process_users(self, user_ids: Optional[List[str]] = None, max_workers: int = 3, start_index: int = 0, generate_english: bool = True, parallel_users: int = 40) -> Dict:
        """处理指定用户列表（如果为None则处理所有用户）
        
        Args:
            user_ids: 要处理的用户ID列表
            max_workers: 单个用户内部的并行工作线程数（用于图片生成等）
            start_index: 起始序号（用于显示和目录命名）
            generate_english: 是否自动生成英文版（默认True）
            parallel_users: 并行处理的用户数量（默认40个用户同时处理）
        """
        if user_ids is None:
            user_ids = self.get_available_users()
            print(f"自动检测到 {len(user_ids)} 个用户: {user_ids}")

        results = {}
        all_available_users = self.get_available_users()
        
        print(f"\n{'='*60}")
        print(f"🚀 并行处理模式: 最多 {parallel_users} 个用户同时处理")
        print(f"   单个用户内部并发数: {max_workers}")
        print(f"{'='*60}\n")
        
        # 使用 ThreadPoolExecutor 并行处理多个用户
        with ThreadPoolExecutor(max_workers=parallel_users) as executor:
            # 提交所有任务
            future_to_user = {}
            for user_id in user_ids:
                # 获取用户在总列表中的序号
                if user_id in all_available_users:
                    user_index = all_available_users.index(user_id)
                else:
                    user_index = None
                
                future = executor.submit(
                    self._process_user_wrapper,
                    user_id,
                    max_workers,
                    user_index,
                    generate_english
                )
                future_to_user[future] = user_id
            
            # 处理完成的任务
            completed_count = 0
            total_count = len(user_ids)
            
            for future in as_completed(future_to_user):
                user_id = future_to_user[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    if result:
                        results[user_id] = result
                        print(f"\n✅ [{completed_count}/{total_count}] 用户 {user_id} 处理完成")
                    else:
                        print(f"\n⚠️  [{completed_count}/{total_count}] 用户 {user_id} 未生成结果")
                except Exception as e:
                    print(f"\n❌ [{completed_count}/{total_count}] 处理用户 {user_id} 时出错: {e}")
                    import traceback
                    traceback.print_exc()

        # 生成汇总报告
        if results:
            self._generate_summary(results)

        return results
    
    def _process_user_wrapper(self, user_id: str, max_workers: int, user_index: Optional[int], generate_english: bool) -> Optional[Dict]:
        """包装函数，用于在线程池中执行 process_user
        
        这个包装函数确保每个用户的处理过程独立运行，不会相互干扰
        """
        try:
            return self.process_user(user_id, max_workers, user_index=user_index, generate_english=generate_english)
        except Exception as e:
            print(f"用户 {user_id} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_summary(self, results: Dict):
        """生成处理汇总报告，追加模式"""
        summary_path = os.path.join(self.output_base, "processing_summary.json")

        # 读取现有的汇总数据（如果存在）
        existing_summary = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    existing_summary = json.load(f)
            except:
                existing_summary = {}

        # 获取当前时间戳
        from datetime import datetime
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 更新汇总数据
        if "processing_sessions" not in existing_summary:
            existing_summary["processing_sessions"] = []

        # 创建当前会话记录
        session_id = f"session_{len(existing_summary['processing_sessions']) + 1}"
        current_session = {
            "session_id": session_id,
            "timestamp": current_time,
            "total_users_processed": len(results),
            "user_results": {}
        }

        for user_id, result in results.items():
            post_data = result.get("personalized_post", {})
            text_len = len(post_data.get("text", ""))
            img_count = len(post_data.get("images", []))
            
            # 检查是否有英文版
            english_data = result.get("english_post")
            has_english = english_data is not None
            
            current_session["user_results"][user_id] = {
                "post_length": text_len,
                "image_count": img_count,
                "html_path": post_data.get("html_post", ""),
                "user_items_count": result.get("raw_items_count", 0),
                "has_english_version": has_english,
                "english_html_path": english_data.get("html_path", "") if has_english else ""
            }

        # 添加到会话列表
        existing_summary["processing_sessions"].append(current_session)

        # 计算总体统计
        total_sessions = len(existing_summary["processing_sessions"])
        total_users = sum(session["total_users_processed"] for session in existing_summary["processing_sessions"])

        existing_summary["overall_statistics"] = {
            "total_processing_sessions": total_sessions,
            "total_users_processed": total_users,
            "last_updated": current_time
        }

        # 保存汇总报告
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(existing_summary, f, ensure_ascii=False, indent=2)

        print(f"\n📊 汇总报告已更新: {summary_path}")
        print(f"📈 当前会话: {session_id} | 处理用户: {len(results)} 个")
        print(f"📊 累计统计: {total_sessions} 次处理 | {total_users} 个用户")

def resolve_user_selection(inputs: List[str], available_users: List[str]) -> List[str]:
    """辅助函数：将输入的序号或ID转换为真实的用户ID列表
    
    Supports:
    - Single index: "0", "5", "10"
    - Range: "10-20" (inclusive)
    - User ID: direct user ID string
    - Multiple: "0 5 10-15 20"
    """
    selected_users = []
    
    for item in inputs:
        item = item.strip()
        if not item:
            continue
        
        # Check for range format (e.g., "10-20")
        if '-' in item and not item.startswith('-'):
            parts = item.split('-')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start_idx = int(parts[0])
                end_idx = int(parts[1])
                
                # Validate range
                if start_idx > end_idx:
                    print(f"⚠️  警告: 无效范围 [{item}] - 起始值必须 <= 结束值")
                    continue
                
                if start_idx < 0 or end_idx >= len(available_users):
                    print(f"⚠️  警告: 范围 [{item}] 超出边界 (有效范围: 0-{len(available_users)-1})")
                    # Clamp to valid range
                    start_idx = max(0, start_idx)
                    end_idx = min(len(available_users) - 1, end_idx)
                
                # Add all users in range (inclusive)
                for idx in range(start_idx, end_idx + 1):
                    real_id = available_users[idx]
                    if real_id not in selected_users:
                        selected_users.append(real_id)
                
                print(f"✅ 已添加范围 [{start_idx}-{end_idx}]: {end_idx - start_idx + 1} 个用户")
                continue
            
        # Try as single index
        if item.isdigit():
            idx = int(item)
            if 0 <= idx < len(available_users):
                real_id = available_users[idx]
                if real_id not in selected_users:
                    selected_users.append(real_id)
            else:
                print(f"⚠️  警告: 序号 [{idx}] 超出范围 (0-{len(available_users)-1})")
        
        # Try as real ID match (兼容旧习惯)
        elif item in available_users:
            if item not in selected_users:
                selected_users.append(item)
        else:
            print(f"⚠️  警告: 未找到序号或用户ID: {item}")
            
    return selected_users

def main():
    """主函数 - 支持序号选择用户"""
    import argparse

    parser = argparse.ArgumentParser(description='小红书用户数据分析与内容生成')
    parser.add_argument('--users', nargs='+', help='指定要处理的用户序号、范围或ID（如：0 5-10 15）')
    parser.add_argument('--all', action='store_true', help='处理所有可用用户')
    parser.add_argument('--workers', type=int, default=4, help='单个用户内部的并行工作线程数（用于图片生成等）')
    parser.add_argument('--parallel', type=int, default=40, help='并行处理的用户数量（默认：40）')
    parser.add_argument('--no-english', action='store_true', help='不生成英文版（默认：自动生成）')
    parser.add_argument('--max-reflections', type=int, default=None, help='最大反思轮数（默认：3，可通过环境变量MAX_REFLECTION_ITERATIONS设置）')

    args = parser.parse_args()

    # 创建处理器
    processor = RedBookProcessor()
    
    # 获取可用用户
    available_users = processor.get_available_users()
    if not available_users:
        print("❌ 未找到任何用户数据，请检查 download/redbook 目录结构")
        return

    processor.setup_environment()
    
    # 设置反思轮数（优先使用命令行参数，否则交互式输入）
    if args.max_reflections is not None:
        max_reflections = args.max_reflections
        os.environ["MAX_REFLECTION_ITERATIONS"] = str(max_reflections)
        print(f"✅ 反思轮数已设置为: {max_reflections}")
    else:
        # 交互式输入反思轮数
        try:
            default_reflections = int(os.getenv("MAX_REFLECTION_ITERATIONS", "3"))
            print(f"\n💡 反思轮数设置（默认: {default_reflections}）")
            reflection_input = input(f"请输入最大反思轮数（直接回车使用默认值 {default_reflections}）: ").strip()
            if reflection_input:
                max_reflections = int(reflection_input)
                os.environ["MAX_REFLECTION_ITERATIONS"] = str(max_reflections)
                print(f"✅ 反思轮数已设置为: {max_reflections}")
            else:
                max_reflections = default_reflections
                print(f"✅ 使用默认反思轮数: {max_reflections}")
        except ValueError:
            print(f"⚠️  输入无效，使用默认值: {default_reflections}")
            max_reflections = default_reflections
        except KeyboardInterrupt:
            print("\n⚠️  用户取消输入，使用默认值")
            max_reflections = default_reflections

    # 打印用户映射表
    print(f"\n{'='*20} 可用用户列表 {'='*20}")
    print(f"{'序号':<6} | {'用户ID'}")
    print("-" * 40)
    for idx, user_id in enumerate(available_users):
        print(f"[{idx:<4}] : {user_id}")
    print("-" * 40)

    target_user_ids = []

    # 确定要处理的用户
    if args.all:
        print("🚀 已选择处理所有用户")
        target_user_ids = available_users
        
    elif args.users:
        # 命令行参数传入 (可能是序号，也可能是ID)
        target_user_ids = resolve_user_selection(args.users, available_users)
        
    else:
        # 交互式选择
        print(f"\n共找到 {len(available_users)} 个用户。")
        print("💡 提示: 可以使用范围（如 '0-5' 或 '10-20'）或单个序号（如 '0 5 10'）")
        user_input = input("请输入要处理的【序号/范围】(多个用空格分隔，输入 'all' 处理所有): ")

        if user_input.strip().lower() == 'all':
            target_user_ids = available_users
        else:
            target_user_ids = resolve_user_selection(user_input.split(), available_users)

    # 最终确认
    if not target_user_ids:
        print("❌ 未选择有效的用户，程序退出")
        return

    print(f"\n✅ 即将处理以下 {len(target_user_ids)} 个用户:")
    for uid in target_user_ids:
        # 反向查找序号以便显示
        idx = available_users.index(uid)
        print(f"  [{idx}] {uid}")
    print(f"{'='*50}\n")

    # 开始处理
    results = processor.process_users(
        target_user_ids, 
        max_workers=args.workers, 
        generate_english=not args.no_english,
        parallel_users=args.parallel
    )

    if results:
        print(f"\n🎉 处理完成！共生成 {len(results)} 个用户的个性化内容")
        for user_id in results.keys():
            idx = available_users.index(user_id) if user_id in available_users else None
            dir_name = f"{idx}_{user_id}" if idx is not None else user_id
            print(f"  用户 {user_id}: 查看 generated_it/{dir_name}/social_media_post.html")
    else:
        print("❌ 未处理任何用户数据")

if __name__ == "__main__":
    main()
