import base64
import os
import json
import requests
import urllib.parse
import re  # <--- 新增正则库，用于处理 Markdown
from datetime import datetime
from pathlib import Path

# Handle imports that work both from main.py (absolute) and from agents/ directory (relative)
try:
    # Try absolute import first (when running from main.py)
    from agents.rag_embedding_helper import RAGEmbeddingHelper
    from agents.html_parser_for_reflection import HTMLParserForReflection
    from agents.reflection_advisor import ReflectionAdvisor
    from agents.evaluate_groupscore import evaluate_file, CLIPEvaluator
except ImportError:
    # Fallback to relative import (when running from agents/ directory)
    from rag_embedding_helper import RAGEmbeddingHelper
    from html_parser_for_reflection import HTMLParserForReflection
    from reflection_advisor import ReflectionAdvisor
    from evaluate_groupscore import evaluate_file, CLIPEvaluator

# 尝试加载 .env 文件（如果还没有加载）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv 未安装，跳过


class AIRefusalError(Exception):
    """AI连续拒绝生成内容的异常"""
    pass


class ITProductGenerator:
    def __init__(self, ideas, examples_dir="agents/redbook", enable_links=False):
        # 文本生成模型
        self.chat_api_key = os.getenv("CHAT_API_KEY")
        self.chat_base_url = os.getenv("CHAT_BASE_URL")
        self.chat_model = os.getenv("CHAT_MODEL")

        # 图像生成模型
        self.generate_api_key = os.getenv("IMAGE_API_KEY")
        self.generate_base_url = os.getenv("IMAGE_BASE_URL")
        self.generate_model = os.getenv("IMAGE_MODEL")
        
        # 链接生成开关（默认关闭）
        self.enable_links = enable_links
        
        # 联网搜索模型（用于获取真实链接）
        self.search_api_key = os.getenv("SEARCH_API_KEY")
        self.search_base_url = os.getenv("SEARCH_BASE_URL", "https://yunwu.ai/v1")
        self.search_model = os.getenv("SEARCH_MODEL", "gpt-5-all")
        
        # 打印链接生成状态
        if self.enable_links:
            print(f"🔗 链接生成功能: ✅ 已启用")
            # 调试信息：检查环境变量
            if not self.search_api_key:
                print(f"   ⚠️  警告: SEARCH_API_KEY未设置，链接生成可能失败")
                # 检查是否有其他可能的 API key 名称
                alternative_keys = ["GENERATE_API_KEY", "IMAGE_API_KEY", "CHAT_API_KEY"]
                found_alternatives = []
                for key in alternative_keys:
                    if os.getenv(key):
                        found_alternatives.append(key)
                if found_alternatives:
                    print(f"   💡 提示: 找到了其他 API key: {', '.join(found_alternatives)}")
                    print(f"      💡 但需要的是 SEARCH_API_KEY，请检查 .env 文件")
        else:
            print(f"🔗 链接生成功能: ⚠️  已禁用 (enable_links=False)")
        
        # 创意生成器
        self.ideas = ideas
        
        # 样例目录
        self.examples_dir = examples_dir
        
        # RAG settings
        self.rag_enabled = True  # Enable RAG-based example retrieval
        self.dataset_name = "redbook"  # Dataset name for cache identification
        self.data_root = os.path.join(os.path.dirname(__file__), "..", "download", "redbook")
        self._rag_cache = {}  # Cache for RAG retrieval results (key: user_id)
        
        # Initialize RAG embedding helper
        self.rag_helper = RAGEmbeddingHelper(
            api_key=self.search_api_key,
            api_base=self.search_base_url,
            cache_dir=os.path.join(os.path.dirname(__file__), "..", "embeddings_cache")
        )
        
        # 加载小红书样例（将改为支持 RAG）
        self.examples = []  # 初始化为空，将在需要时动态加载
        
        # Reflection mechanism settings
        self.reflection_enabled = True  # Enable reflection by default
        self.reflection_threshold = float(os.getenv("REFLECTION_THRESHOLD_IT", "0.65"))
        # Maximum reflection iterations (can be configured via environment variable)
        # First 3 iterations use specific strategies, iterations >= 3 all use iteration 2's strategy
        self.max_reflection_iterations = int(os.getenv("MAX_REFLECTION_ITERATIONS", "3"))
        self.reflection_strict_mode = os.getenv("REFLECTION_STRICT_MODE", "true").lower() == "true"  # 严格模式：只接受提升score的修改
        
        # Initialize reflection components
        try:
            self.html_parser = HTMLParserForReflection()
            self.reflection_advisor = ReflectionAdvisor()
            
            # Initialize CLIP evaluator for GroupScore calculation
            clip_model = os.getenv("CLIP_MODEL", "ViT-B/32")
            clip_device = os.getenv("CLIP_DEVICE", "cuda")
            self.clip_evaluator = CLIPEvaluator(model_name=clip_model, device=clip_device)
            
            strict_mode_status = "✅ 开启" if self.reflection_strict_mode else "⚠️  关闭"
            print(f"✅ Reflection机制已启用 (阈值: {self.reflection_threshold}, 最多{self.max_reflection_iterations}次迭代, 严格模式: {strict_mode_status})")
            if self.max_reflection_iterations > 3:
                print(f"   ℹ️  前3次使用特定策略，第4-{self.max_reflection_iterations}次重复使用第3次策略")
        except Exception as e:
            print(f"⚠️  Reflection机制初始化失败: {e}, 将跳过reflection")
            self.reflection_enabled = False
    
    def extract_top1_preference(self, profile_data):
        """
        Extract top1 preference from user profile
        
        Args:
            profile_data: User profile dict or string
            
        Returns:
            String containing the top1 preference description
        """
        if isinstance(profile_data, dict):
            profile_text = profile_data.get("profile_text", json.dumps(profile_data, ensure_ascii=False))
        else:
            profile_text = str(profile_data)
        
        # Try to extract "1. Preference 1:" or "Preference 1:" pattern
        patterns = [
            r'1\.\s*Preference\s*1:\s*([^\n]+(?:\n\s+Reason:[^\n]+)?)',
            r'Preference\s*1:\s*([^\n]+(?:\n\s+Reason:[^\n]+)?)',
            r'1\.\s*([^\n]+(?:\n\s+Reason:[^\n]+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, profile_text, re.IGNORECASE)
            if match:
                preference = match.group(1).strip()
                print(f"📋 Extracted Top1 Preference: {preference[:100]}...")
                return preference
        
        # Fallback: return first 500 chars of profile
        print(f"⚠️ Could not extract structured preference, using profile preview")
        return profile_text[:500]
    
    def extract_user_id_from_path(self, file_path):
        """
        Extract user ID from file path
        
        Args:
            file_path: Path to user profile or output directory
            
        Returns:
            User ID string, or None if not found
        """
        # Try to extract from path pattern like: generated_it/0_5435e123d6e4a965e190095a/
        # or download/redbook/{user_id}/
        
        # Pattern 1: {number}_{alphanumeric_id}
        match = re.search(r'[\\/](\d+_[a-f0-9]+)[\\/]', file_path)
        if match:
            full_id = match.group(1)
            # Extract the alphanumeric part after underscore
            user_id = full_id.split('_', 1)[1] if '_' in full_id else full_id
            print(f"📌 Extracted user_id from path: {user_id}")
            return user_id
        
        # Pattern 2: pure numeric ID
        match = re.search(r'[\\/](\d{10,})[\\/]', file_path)
        if match:
            user_id = match.group(1)
            print(f"📌 Extracted user_id from path: {user_id}")
            return user_id
        
        print(f"⚠️ Could not extract user_id from path: {file_path}")
        return None
    
    def load_examples_with_rag(self, user_id, top1_preference, top_k=3):
        """
        Load examples using RAG (Retrieval-Augmented Generation)
        Retrieves most relevant examples from user's historical and recommended posts
        
        Args:
            user_id: User ID
            top1_preference: Top1 preference text for similarity matching
            top_k: Number of examples to retrieve
            
        Returns:
            List of example dicts with content
        """
        # Check cache
        cache_key = f"{user_id}_{top_k}"
        if cache_key in self._rag_cache:
            print(f"✅ Using cached RAG examples ({len(self._rag_cache[cache_key])} posts)")
            return self._rag_cache[cache_key]
        
        print(f"🔍 RAG Mode: Retrieving relevant examples for user {user_id}...")
        
        try:
            # Step 1: Build embeddings for user files
            print(f"📊 Building embeddings for user files...")
            embeddings_data = self.rag_helper.build_embeddings_for_user(
                dataset_name=self.dataset_name,
                dataset_root=self.data_root,
                user_id=user_id,
                max_workers=10,
                use_cache=True
            )
            
            if not embeddings_data["embeddings"]:
                print(f"⚠️ No embeddings found, falling back to default examples")
                return self._load_examples_fallback()
            
            # Step 2: Retrieve top-k similar files
            print(f"🎯 Retrieving top-{top_k} similar examples based on preference...")
            similar_files = self.rag_helper.retrieve_top_k_similar(
                query_text=top1_preference,
                embeddings_data=embeddings_data,
                top_k=top_k
            )
            
            if not similar_files:
                print(f"⚠️ No similar files found, falling back to default examples")
                return self._load_examples_fallback()
            
            # Step 3: Load file contents and parse (including images)
            examples = []
            for i, file_info in enumerate(similar_files):
                try:
                    content = self.rag_helper.get_file_content(file_info["path"])
                    
                    if not content:
                        continue
                    
                    # Add text content
                    examples.append({
                        "type": "text",
                        "content": content,
                        "filename": file_info["filename"],
                        "similarity": file_info["similarity"],
                        "folder": file_info["folder"],
                        "post_id": file_info["post_id"]
                    })
                    
                    print(f"   ✅ Retrieved: {file_info['filename']} (similarity: {file_info['similarity']:.3f})")
                    
                    # Try to load the first image from this post as a style reference
                    try:
                        # Get the post directory (parent of note.txt)
                        post_dir = Path(file_info["path"]).parent
                        images_dir = post_dir / "images"
                        
                        if images_dir.exists():
                            # Find the first image
                            image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.webp"))
                            if image_files:
                                first_image = image_files[0]
                                with open(first_image, 'rb') as img_f:
                                    image_data = img_f.read()
                                    base64_image = base64.b64encode(image_data).decode('utf-8')
                                    
                                    # Determine image format
                                    ext = first_image.suffix.lower().strip('.')
                                    if ext == 'jpg':
                                        ext = 'jpeg'
                                    
                                    examples.append({
                                        "type": "image",
                                        "content": base64_image,
                                        "format": ext,
                                        "filename": first_image.name,
                                        "post_id": file_info["post_id"],
                                        "similarity": file_info["similarity"]
                                    })
                                    print(f"      📸 Loaded image: {first_image.name}")
                    except Exception as img_err:
                        # Image loading is optional, don't fail the whole process
                        pass
                        
                except Exception as e:
                    print(f"   ⚠️ Failed to load {file_info['path']}: {e}")
            
            # Cache the results
            self._rag_cache[cache_key] = examples
            
            # Count text and image examples
            text_count = sum(1 for e in examples if e["type"] == "text")
            image_count = sum(1 for e in examples if e["type"] == "image")
            print(f"💾 Cached {len(examples)} RAG example(s): {text_count} 文本, {image_count} 图片")
            
            return examples
            
        except Exception as e:
            print(f"⚠️ RAG retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            return self._load_examples_fallback()
    
    def _load_examples_fallback(self):
        """
        Fallback method: Load examples from fixed directory (agents/redbook/)
        This is the original _load_examples logic
        """
        examples = []
        
        if not os.path.exists(self.examples_dir):
            print(f"⚠️ 样例目录不存在: {self.examples_dir}，将不使用样例")
            return examples
        
        try:
            # 遍历目录中的文件
            for filename in os.listdir(self.examples_dir):
                filepath = os.path.join(self.examples_dir, filename)
                
                # 处理文本文件
                if filename.endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text_content = f.read().strip()
                        if text_content:
                            examples.append({
                                "type": "text",
                                "content": text_content,
                                "filename": filename
                            })
                
                # 处理图片文件
                elif filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    with open(filepath, 'rb') as f:
                        image_data = f.read()
                        base64_image = base64.b64encode(image_data).decode('utf-8')
                        
                        # 确定图片格式
                        ext = filename.lower().split('.')[-1]
                        if ext == 'jpg':
                            ext = 'jpeg'
                        
                        examples.append({
                            "type": "image",
                            "content": base64_image,
                            "format": ext,
                            "filename": filename
                        })
            
            if examples:
                text_count = sum(1 for e in examples if e["type"] == "text")
                image_count = sum(1 for e in examples if e["type"] == "image")
                print(f"✅ 加载了RAG {len(examples)} 个样例: {text_count} 个文本, {image_count} 张图片")
            else:
                print(f"⚠️ {self.examples_dir} 目录下没有找到样例文件")
        
        except Exception as e:
            print(f"⚠️ 加载样例时出错: {e}")
        
        return examples
    
    def _load_examples(self, user_profile_path=None, profile_data=None):
        """
        Load example posts - supports both RAG mode and fallback mode
        
        Args:
            user_profile_path: Path to user profile (for extracting user_id)
            profile_data: User profile data (for extracting top1 preference)
            
        Returns:
            List of example dicts with content
        """
        # If RAG is enabled and we have necessary info, use RAG
        if self.rag_enabled and user_profile_path and profile_data:
            try:
                # Extract user_id from path
                user_id = self.extract_user_id_from_path(user_profile_path)
                
                if user_id:
                    # Extract top1 preference
                    top1_preference = self.extract_top1_preference(profile_data)
                    
                    # Use RAG to retrieve examples (top-3 for consistency with reflection)
                    examples = self.load_examples_with_rag(user_id, top1_preference, top_k=3)
                    
                    if examples:
                        return examples
                    else:
                        print("⚠️ RAG returned no examples, using fallback")
                else:
                    print("⚠️ Could not extract user_id, using fallback")
            except Exception as e:
                print(f"⚠️ RAG mode failed: {e}, using fallback")
                import traceback
                traceback.print_exc()
        
        # Fallback: use fixed examples
        return self._load_examples_fallback()

    def _is_ai_refusal(self, text):
        """
        检测AI是否拒绝生成内容
        
        Args:
            text: AI返回的文本
            
        Returns:
            bool: 如果是拒绝响应返回True
        """
        if not text or len(text) < 10:
            return True
        
        # 常见的拒绝响应模式
        refusal_patterns = [
            "I'm sorry, I can't assist with that",
            "I cannot assist with that",
            "I can't help with that",
            "I'm unable to assist",
            "I cannot provide",
            "I'm sorry, but I can't",
            "I apologize, but I cannot",
            "抱歉，我无法协助",
            "抱歉，我不能",
            "对不起，我无法",
        ]
        
        text_lower = text.lower().strip()
        
        for pattern in refusal_patterns:
            if pattern.lower() in text_lower:
                return True
        
        # 检测过短的响应（通常是拒绝）
        if len(text) < 50:
            return True
        
        return False
    
    def generate_text(self, user_profile, user_profile_path=None, profile_data=None):
        """根据用户画像生成文案，支持 RAG 模式动态加载样例
        
        Args:
            user_profile: User profile text
            user_profile_path: Path to user profile (for RAG)
            profile_data: Profile data dict (for RAG)
        """
        
        # 加载样例（支持 RAG 模式）
        examples = self._load_examples(user_profile_path=user_profile_path, profile_data=profile_data)
        
        # 构建创意提示
        idea_prompt = f"""
        **创意指导：**
        {json.dumps(self.ideas, ensure_ascii=False, indent=2)}
        """
        
        # 构建样例提示（如果有样例）
        examples_prompt = ""
        image_examples = []
        
        if examples:
            examples_prompt = "\n**参考样例：**\n"
            
            # Extract text examples (只保留1个，且更短)
            text_examples = [e for e in examples if e["type"] == "text"]
            if text_examples:
                examples_prompt += f"\n样例:\n{text_examples[0]['content'][:200]}\n"  # 只保留第一个，且只200字
            
            # Extract image examples for multimodal reference (保留但简化说明)
            image_examples = [e for e in examples if e["type"] == "image"]
            if image_examples:
                examples_prompt += f"\n**配图样例：已提供 {len(image_examples)} 张参考图片**\n"
        
        # Build prompt text
        prompt_text = f"""
            你是一位活跃在小红书平台的真实博主，需要根据用户画像和创意指导创作一篇小红书风格的图文帖子。    

            **用户画像特征：**
            {user_profile}

            {idea_prompt}
            
            {examples_prompt}
            
            ⭐ **核心创作原则**：
            - 像真人一样写作，表达要有变化和多样性
            - 避免AI式的重复用词（如反复说"绝绝子"、"宝子们"）
            - 同样的意思用不同的表达方式
            - 保持自然、真诚、有个性
            
            **创作要求：**
            一、格式要求：
            1. **标签（Tags）- 必须首先输出**：
               - 在正文开始之前，先输出 1-4 个小红书风格的标签
               - 格式：在第一行输入 "===TAGS===" 后换行，每个标签单独一行
               - 标签要贴近小红书真实风格，例如：
                 * 美食类：美食探店、火锅、成都美食、人均100以下、川菜
                 * 旅行类：旅行vlog、杭州旅游、周末去哪玩、江南水乡、小众景点
                 * 好物类：好物分享、数码测评、iPhone、性价比之选、科技好物
                 * 生活类：日常生活、周末日记、咖啡店、氛围感、松弛感生活
               - 标签要具体，能体现内容核心，避免过于宽泛
               - 输出完tags后换行输入 "===CONTENT===" 再开始正文
            
            2. **正文**：语言自然、细节丰富、口语化（中文），300-800字为宜，但上不设限。
            
            3. **排版要求（重要）**：
               - 请适当使用 **加粗** (markdown语法) 来标记关键词或重点，这能提升阅读体验。
               - 段落之间要分明，善用emoji分隔或装饰（如✨🔥💕等）。
               - 不要使用一级或二级标题（# 或 ##），使用emoji或加粗来引导小节。
            
            4. **链接推荐**：
               - 在正文结束后，换行输入 "===LINKS==="。
               - 推荐 2 个与帖子内容高度相关的**国内平台**延伸阅读内容的搜索关键词。
               - 格式必须为：**推荐语/标题 | 平台名称 | 搜索关键词**
               - 平台仅限：**小红书、B站、知乎、抖音、微博**。
               - 搜索关键词要精确（如"iPhone 15 Pro 测评"而非"手机测评"）。
            
            二、内容要求（小红书风格）：
            0. **文本形式**：视创意和画像而定，可以是种草分享、探店日记、教程攻略、好物推荐、旅行vlog等小红书常见形式
            
            1. **语言风格 - 贴近小红书用户（注意多样化表达）**：
               
               📢 **称呼方式（换着用，不要总用同一个）**：
               - 开头称呼：姐妹们 / 宝子们 / 集美们 / 家人们 / 朋友们 / 大家
               - 或者直接开门见山，不用称呼
               
               ✨ **表达赞美（避免重复"绝绝子"）**：
               - 超级好：绝了 / 太棒了 / 爱了爱了 / 真香 / 无敌了 / 太可了 / 好爱 / 太赞了
               - 很推荐：yyds / 强推 / 必冲 / 值得 / 不踩雷 / 闭眼入 / 可以试试 / 真心推荐
               - 惊艳：惊艳到我了 / 太美了 / 被圈粉了 / 沦陷了 / 上头了 / 欲罢不能
               
               💕 **表达感受（自然真实）**：
               - 好评：好吃哭了 / 爱不释手 / 念念不忘 / 回购无数次 / 终于找到了
               - 氛围感：松弛感满满 / 氛围感拉满 / 治愈系 / 很chill / 很舒服 / 岁月静好
               - 情绪：谁懂啊 / 破防了 / emo了 / 有被感动到 / 太治愈了
               
               🎯 **推荐表达（灵活运用）**：
               - 推荐：种草了 / 安利给你们 / 分享给你们 / 墙裂推荐 / 赶紧冲 / 值得拥有
               - 劝退：避雷 / 不推荐 / 踩雷了 / 翻车了 / 别买 / 慎入
               
               🔥 **网络用语（适度使用，不要堆砌）**：
               - 程度词：狠狠地 / 疯狂 / 无限 / 死磕
               - 状态词：DNA动了 / 破防 / 上头 / 沦陷 / 破大防
               - 形容词：绝美 / 奈斯 / 巨好 / 太对了 / 就是它
               
               💬 **语言技巧**：
               - 口语化、年轻化，使用感叹号表达热情！
               - 适当使用emoji（但不要过度）
               - 多用短句，节奏轻快
               - 偶尔用疑问句增加互动感
               - 避免同一词汇反复出现（如连续3次"绝绝子"）
               - 保持真诚，不要刻意堆砌网络用语
               
            2. **情感真实、热情分享**：
               - 表达真实的情绪，可以用"真的超好吃！"、"我爱了！"、"强烈推荐！"等
               - 像给朋友安利一样的语气
            
            3. **结构自由但要有重点**：
               - 开头抓眼球（可以直接说结论，如"这家店我要吹爆！"）
               - 中间详细介绍
               - 结尾可以互动（如"你们去过吗？"、"评论区说说你的最爱"）
            
            **避免以下AI常见问题：**
            - ❌ 不要使用严肃的议论文口吻
            - ❌ 不要过度堆砌网络用语，保持自然
            - ❌ 不要写成广告软文，要像真实分享
            - ❌ **不要重复使用同样的词汇**（如不要3次都说"绝绝子"，要换着用"绝了"、"太棒了"、"爱了"等）
            - ❌ **不要每段开头都用同样的称呼**（如不要段段都是"宝子们"）
            - ✅ 保持表达的多样性和自然性，像真人写作一样有变化

            请开始创作小红书风格的图文帖子：
            """
        
        # Build message content (support multimodal if images are provided)
        message_content = []
        
        # Add image examples first (if available) for visual style reference
        if image_examples:
            print(f"   📸 提供 {len(image_examples)} 张图片样例作为视觉风格参考")
            for i, img_ex in enumerate(image_examples[:2], 1):  # Max 2 images to avoid token limits
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{img_ex['format']};base64,{img_ex['content']}"
                    }
                })
            message_content.append({
                "type": "text",
                "text": f"以上是 {len(image_examples[:2])} 张小红书真实帖子的配图样例，请参考它们的视觉风格、构图和色调。\n\n"
            })
        
        # Add main prompt text
        message_content.append({
            "type": "text",
            "text": prompt_text
        })
        
        # Make API call with retry mechanism for refusal responses
        max_retries = 3
        full_content = None
        
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.chat_base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.chat_api_key}"},
                    json={
                        "model": self.chat_model,
                        "messages": [{"role": "user", "content": message_content}],
                        "temperature": 0.3 + (attempt - 1) * 0.05,  # 低temperature让初始生成更简单基础
                        "top_p": 0.9,
                    },
                    timeout=60
                )
                
                if resp.status_code != 200:
                    print(f"   ⚠️  API错误 {resp.status_code}: {resp.text[:200]}")
                    if attempt < max_retries:
                        print(f"   🔄 重试 {attempt + 1}/{max_retries}...")
                        import time
                        time.sleep(2)
                        continue
                    else:
                        raise Exception(f"API调用失败: {resp.status_code}")
                
                full_content = resp.json()["choices"][0]["message"]["content"].strip()
                
                # 检测AI是否拒绝生成
                if self._is_ai_refusal(full_content):
                    print(f"   ⚠️  检测到AI拒绝响应（尝试 {attempt}/{max_retries}）")
                    print(f"       内容: {full_content[:100]}...")
                    
                    if attempt < max_retries:
                        print(f"   🔄 重试生成...")
                        import time
                        time.sleep(1)
                        continue
                    else:
                        print(f"   ❌ {max_retries}次尝试后仍然被拒绝")
                        print(f"   💡 跳过该用户...")
                        raise AIRefusalError(f"AI连续{max_retries}次拒绝生成内容")
                else:
                    # 成功生成正常内容
                    if attempt > 1:
                        print(f"   ✅ 第 {attempt} 次尝试成功生成内容")
                    break
                    
            except Exception as e:
                print(f"   ❌ 生成文案异常 (尝试 {attempt}/{max_retries}): {str(e)[:100]}")
                if attempt < max_retries:
                    import time
                    time.sleep(2)
                else:
                    raise
        
        # 确保有内容
        if not full_content:
            raise Exception("生成文案失败：AI未返回有效内容")
        
        # --- 解析标签、正文和链接 ---
        tags = []
        text_body = full_content
        links = []
        
        # 1. 提取标签
        if "===TAGS===" in full_content:
            parts = full_content.split("===TAGS===", 1)
            remaining = parts[1]
            
            if "===CONTENT===" in remaining:
                tag_section, remaining = remaining.split("===CONTENT===", 1)
                # 解析标签
                for line in tag_section.strip().split('\n'):
                    tag = line.strip()
                    if tag:
                        tags.append(tag)
                
                text_body = remaining
            else:
                # 没有 ===CONTENT===，直接从 ===TAGS=== 后面找正文
                text_body = remaining
        
        # 2. 提取链接（先提取关键词）
        link_keywords = []
        if "===LINKS===" in text_body:
            parts = text_body.split("===LINKS===")
            text_body = parts[0].strip()
            link_section = parts[1].strip()
            
            # 解析每一行链接，先收集关键词
            for line in link_section.split('\n'):
                line = line.strip()
                if '|' in line:
                    try:
                        segments = [s.strip() for s in line.split('|')]
                        if len(segments) >= 3:
                            # 清洗标题中的特殊符号
                            raw_title = segments[0]
                            clean_title = raw_title.replace('**', '').replace('__', '').replace('*', '')
                            clean_title = re.sub(r'^[\-\•\d\.\s]+', '', clean_title).strip()

                            platform = segments[1]
                            keyword = segments[2]
                            
                            link_keywords.append({
                                "title": clean_title,
                                "platform": platform,
                                "keyword": keyword
                            })
                    except Exception as e:
                        print(f"解析链接行出错: {line} - {e}")
            
            # 保持 links 为空列表，稍后根据搜索结果填充
            # 不要将 link_keywords 赋值给 links，避免共享引用
        
        # 3. 如果有链接关键词，使用联网搜索获取真实链接（带重试机制）
        if link_keywords and self.enable_links:
            print(f"🔍 使用联网搜索获取真实帖子链接（最多重试2次）...")
            real_links = self._search_real_links_with_retry(link_keywords, text_body, max_retries=2)
            if real_links:
                links = real_links
                print(f"✅ 成功获取 {len(links)} 个真实链接")
            else:
                print(f"⚠️ 联网搜索失败，降级到搜索链接")
                # 降级：使用关键词生成搜索链接
                links = []
                for link_kw in link_keywords:
                    search_url = self._generate_search_url(link_kw['platform'], link_kw['keyword'])
                    links.append({
                        "title": link_kw['title'],
                        "platform": link_kw['platform'],
                        "url": search_url
                    })
                print(f"   ✅ 已生成 {len(links)} 个搜索链接作为备选")
        elif link_keywords and not self.enable_links:
            print(f"ℹ️  链接生成功能已禁用（enable_links=False），跳过 {len(link_keywords)} 个链接关键词")
        
        return text_body, tags, links
    
    def _search_real_links_with_retry(self, link_keywords, text_content, max_retries=2):
        """
        使用联网搜索获取真实的帖子链接（带重试机制）
        
        Args:
            link_keywords: 链接关键词列表
            text_content: 文本内容
            max_retries: 最大重试次数（默认2次）
        
        Returns:
            真实链接列表，如果所有重试都失败则返回空列表
        """
        import time
        
        for attempt in range(1, max_retries + 1):
            print(f"\n🔄 尝试 {attempt}/{max_retries} 获取真实链接...")
            
            real_links = self._search_real_links(link_keywords, text_content)
            
            if real_links and len(real_links) > 0:
                print(f"✅ 第 {attempt} 次尝试成功，获取到 {len(real_links)} 个真实链接")
                return real_links
            else:
                if attempt < max_retries:
                    wait_time = 1  # 等待时间：1秒
                    print(f"⚠️ 第 {attempt} 次尝试未找到真实链接，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ 所有 {max_retries} 次尝试都未找到真实链接")
        
        return []
    
    def _search_real_links(self, link_keywords, text_content):
        """
        使用联网搜索获取真实的帖子链接（单次尝试）
        参考 commentproduct_generator.py 的实现
        """
        if not self.search_api_key:
            # 添加调试信息
            env_value = os.getenv("SEARCH_API_KEY")
            if env_value is None:
                print("⚠️ SEARCH_API_KEY not found in environment variables")
                print("   💡 请检查 .env 文件或环境变量中是否设置了 SEARCH_API_KEY")
            elif env_value == "":
                print("⚠️ SEARCH_API_KEY is empty string")
                print("   💡 请检查 .env 文件中 SEARCH_API_KEY 的值是否正确")
            else:
                print(f"⚠️ SEARCH_API_KEY exists but is falsy (value length: {len(env_value)})")
            print("   ⚠️ Fallback to search URLs")
            return None
        
        # 构建搜索提示
        links_desc = []
        for i, link in enumerate(link_keywords):
            title = link.get('title', '')
            platform = link.get('platform', '')
            keyword = link.get('keyword', '')
            links_desc.append(f"{i+1}. 平台: {platform}, 关键词: {keyword}, 期望内容: {title}")
        
        search_prompt = f"""基于以下小红书帖子内容和推荐关键词，搜索并返回2个**真实存在的帖子链接**。

帖子内容（前600字）：
{text_content[:600]}

推荐方向：
{chr(10).join(links_desc)}

返回JSON格式（必须是纯JSON数组，不要其他文字）：
[
  {{
    "title": "具体帖子的标题",
    "platform": "平台名称（小红书/B站/知乎/抖音/微博）",
    "url": "真实的帖子URL（必须是具体帖子而非首页或热榜）"
  }}
]

**核心要求（重要）**：
1. **URL 必须是具体的帖子/视频/文章链接，不能是首页或搜索页**
2. **相关性要求宽松**：只要主题相关即可，不需要完全匹配
   - 例如：舞蹈图文可以推荐《只此青绿》相关帖子
   - 例如：美食图文可以推荐同类型美食的帖子
   - 例如：旅行图文可以推荐同目的地的帖子
3. **优先搜索小红书、B站、知乎、抖音、微博等平台的相关内容**
4. **标题要真实完整**
5. **只返回能找到真实URL的链接，如果找不到就返回空数组 []**
6. **不要返回 search_keyword，只返回有真实URL的链接**

**输出要求**：
- 只输出JSON数组，不要任何其他文字
- 如果找不到真实链接，返回空数组 []
- 确保URL是真实可访问的具体帖子链接"""

        try:
            print(f"📡 调用 {self.search_model} 进行联网搜索...")
            response = requests.post(
                f"{self.search_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.search_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.search_model,
                    "web_search_options": {},  # 启用联网搜索
                    "messages": [{"role": "user", "content": search_prompt}],
                    "temperature": 0.7
                },
                timeout=90
            )
            
            if response.status_code != 200:
                print(f"⚠️ 搜索API错误 {response.status_code}")
                return None
            
            resp_json = response.json()
            
            if "choices" not in resp_json:
                print(f"⚠️ 响应格式异常")
                return None
            
            content = resp_json["choices"][0]["message"]["content"]
            
            # 清理 content - 移除引用块和 markdown 链接
            content = re.sub(r'^>.*?$', '', content, flags=re.MULTILINE)
            content = re.sub(r'\*\*\[.*?\]\(.*?\)\*\*\s*·\s*\*.*?\*', '', content)
            content = re.sub(r'\[.*?\]\(.*?\)', '', content)
            content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'^```\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'\n{2,}', '\n', content).strip()
            
            print(f"📄 搜索结果（前500字符）: {content[:500]}...")
            
            # 尝试多种方式提取 JSON
            search_results = None
            
            # 方法1: 尝试直接解析整个 content（可能是纯JSON）
            try:
                parsed = json.loads(content.strip())
                if isinstance(parsed, list):
                    search_results = parsed
                    print("✅ 方法1: 直接解析成功（数组格式）")
                elif isinstance(parsed, dict):
                    # 可能是 {"search_query": [...]} 或其他格式
                    # 尝试查找可能的数组字段
                    for key in ['results', 'links', 'data', 'items']:
                        if key in parsed and isinstance(parsed[key], list):
                            search_results = parsed[key]
                            print(f"✅ 方法1: 从字典中提取数组（字段: {key}）")
                            break
            except:
                pass
            
            # 方法2: 提取 JSON 数组（使用括号匹配）
            if not search_results:
                first_bracket = content.find('[')
                if first_bracket != -1:
                    bracket_count = 0
                    last_bracket = -1
                    in_string = False
                    escape_next = False
                    
                    for i in range(first_bracket, len(content)):
                        char = content[i]
                        if escape_next:
                            escape_next = False
                            continue
                        if char == '\\':
                            escape_next = True
                            continue
                        if char == '"':
                            in_string = not in_string
                            continue
                        if not in_string:
                            if char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    last_bracket = i
                                    break
                    
                    if last_bracket != -1:
                        json_content = content[first_bracket:last_bracket + 1].strip()
                        try:
                            search_results = json.loads(json_content)
                            print("✅ 方法2: 括号匹配提取成功")
                        except:
                            pass
            
            # 方法3: 使用正则表达式查找 JSON 数组
            if not search_results:
                json_pattern = r'\[\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}(?:\s*,\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})*\s*\]'
                match = re.search(json_pattern, content, re.DOTALL)
                if match:
                    try:
                        search_results = json.loads(match.group(0))
                        print("✅ 方法3: 正则表达式提取成功")
                    except:
                        pass
            
            if not search_results:
                print(f"⚠️ 未找到有效的 JSON 数组，原始内容前500字符: {content[:500]}")
                return []
            
            if not isinstance(search_results, list):
                print(f"⚠️ 解析结果不是数组格式: {type(search_results)}")
                return []
            
            # 调试：打印解析后的结果结构
            print(f"🔍 解析后的结果数量: {len(search_results)}")
            if search_results:
                first_result = search_results[0]
                print(f"🔍 第一个结果类型: {type(first_result)}")
                if isinstance(first_result, dict):
                    print(f"🔍 第一个结果字段: {list(first_result.keys())}")
                    print(f"🔍 第一个结果内容: {json.dumps(first_result, ensure_ascii=False, indent=2)[:400]}...")
            
            # 验证和提取有效链接（降低标准，只要真实URL就接受）
            valid_links = []
            if isinstance(search_results, list):
                for idx, result in enumerate(search_results):
                    if not isinstance(result, dict):
                        print(f"⚠️ 结果 {idx} 不是字典格式: {type(result)}")
                        continue
                    
                    # 调试：打印每个结果的字段
                    result_keys = list(result.keys())
                    has_url = 'url' in result_keys
                    print(f"🔍 处理结果 {idx}: 字段={result_keys}, 是否有url字段={has_url}")
                    
                    # 如果没有url字段，打印完整内容以便调试
                    if not has_url:
                        print(f"   ⚠️ 结果 {idx} 没有url字段，完整内容: {json.dumps(result, ensure_ascii=False)[:200]}...")
                    
                    # 只接受有真实 URL 的链接
                    # 尝试多种可能的URL字段名
                    url = None
                    for url_field in ['url', 'link', 'href', 'web_url', 'post_url', 'article_url']:
                        if result.get(url_field):
                            url = result.get(url_field, "").strip()
                            if url:
                                print(f"   ✅ 从字段 '{url_field}' 找到URL: {url[:60]}...")
                                break
                    
                    if url:
                        
                        # 基本验证：必须是 http/https 链接
                        if not (url.startswith("http://") or url.startswith("https://")):
                            continue
                        
                        # 过滤明显的首页和搜索页
                        if self._is_homepage_url(url):
                            print(f"⚠️ 过滤首页链接: {url[:60]}...")
                            continue
                        
                        # 过滤搜索页（关键：不允许搜索链接）
                        if any(indicator in url.lower() for indicator in [
                            '/search?', '/search/', 'search_query=', 'search_result', 
                            '?q=', '&q=', 'keyword=', '&keyword='
                        ]):
                            print(f"⚠️ 过滤搜索页链接: {url[:60]}...")
                            continue
                        
                        # 清理标题中的特殊符号
                        raw_title = result.get("title", "相关帖子")
                        clean_title = raw_title.replace('**', '').replace('__', '').replace('*', '')
                        clean_title = re.sub(r'^[\-\•\d\.\s]+', '', clean_title).strip()
                        if not clean_title:
                            clean_title = "相关帖子"
                        
                        valid_links.append({
                            "title": clean_title,
                            "platform": result.get("platform", "网页"),
                            "url": url
                        })
                        print(f"✅ 找到真实链接: {clean_title[:40]}... -> {url[:60]}...")
            
            if valid_links:
                print(f"✅ 成功获取 {len(valid_links)} 个真实链接")
                return valid_links
            else:
                print(f"⚠️ 本次尝试未找到有效链接")
                print(f"   💡 可能的原因：")
                print(f"      1. API返回的结果中没有 'url' 字段")
                print(f"      2. 所有URL都被过滤（首页/搜索页）")
                print(f"      3. URL格式不正确（不是http/https）")
                return []  # 返回空列表，便于重试机制判断
        
        except Exception as e:
            print(f"⚠️ 联网搜索异常: {e}")
            import traceback
            traceback.print_exc()
            return []  # 返回空列表，便于重试机制判断
    
    def _is_homepage_url(self, url):
        """检测是否为首页链接（需要过滤掉）"""
        if not url or url == "#":
            return True
        
        # 提取路径部分
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.strip('/')
        query = parsed.query
        
        # 如果没有路径或查询参数，可能是首页
        if not path and not query:
            return True
        
        # 如果只有根路径且没有搜索参数
        homepage_patterns = [
            r'^/?$',  # 空路径或只有斜杠
            r'^/?index\.(html?|php)$',  # index 页面
            r'^/?home$',  # home 页面
        ]
        
        for pattern in homepage_patterns:
            if re.match(pattern, path):
                return True
        
        # 检查是否是热榜聚合网站
        hotlist_domains = [
            "tophub.today",
            "remenla.com",
            "shenmehuole.com",
            "imshuai.com",
            "v2hot.com"
        ]
        
        if any(domain in url for domain in hotlist_domains):
            return True
        
        return False

    def _generate_search_url(self, platform, keyword):
        """根据平台和关键词生成真实的搜索链接"""
        kw_encoded = urllib.parse.quote(keyword)
        p = platform.lower()
        
        if "b站" in p or "bilibili" in p:
            return f"https://search.bilibili.com/all?keyword={kw_encoded}"
        elif "小红书" in p:
            return f"https://www.xiaohongshu.com/search_result?keyword={kw_encoded}"
        elif "知乎" in p:
            return f"https://www.zhihu.com/search?type=content&q={kw_encoded}"
        elif "抖音" in p:
            return f"https://www.douyin.com/search/{kw_encoded}"
        elif "微博" in p:
            return f"https://s.weibo.com/weibo?q={kw_encoded}"
        else:
            return f"https://www.baidu.com/s?wd={kw_encoded}"

    def _must_include_people(self, text_content, user_profile):
        """判断是否必须包含人物（只检测非常确定的场景）"""
        text_lower = text_content.lower()
        profile_lower = str(user_profile).lower()
        
        # 只保留绝对确定需要人物的关键词
        must_people_keywords = [
            '自拍', 'ootd', '穿搭', '妆容', '化妆', 
            '发型', '健身', '瑜伽', '舞蹈', '跳舞',
            '试色', '上身', '显瘦', '显高',
            '演唱会', '打卡照'  # 演唱会通常是想拍自己
        ]
        
        for keyword in must_people_keywords:
            if keyword in text_lower or keyword in profile_lower:
                return True
        
        return False
    
    def _must_include_pets(self, text_content, user_profile):
        """判断是否必须包含宠物（只检测非常确定的场景）"""
        text_lower = text_content.lower()
        profile_lower = str(user_profile).lower()
        
        # 只保留绝对确定是宠物内容的关键词
        must_pet_keywords = [
            '我的猫', '我的狗', '我家猫', '我家狗',
            '铲屎官', '毛孩子', '遛狗', '撸猫',
            '猫咪日常', '狗狗日常'
        ]
        
        for keyword in must_pet_keywords:
            if keyword in text_lower or keyword in profile_lower:
                return True
        
        return False
    
    def _get_focus_subject(self, text_content, user_profile):
        """分析内容主题，确定图片聚焦对象"""
        text_lower = text_content.lower()
        profile_lower = str(user_profile).lower()
        combined = text_lower + ' ' + profile_lower
        
        # 分析主题
        if any(kw in combined for kw in ['美食', '餐厅', '火锅', '烧烤', '咖啡', '奶茶', '甜品', '料理', '菜']):
            return "美食", "食物本身（特写、摆盘、质感）"
        
        elif any(kw in combined for kw in ['旅行', '旅游', '景点', '风景', '建筑', '寺庙', '公园', '海边', '山']):
            return "旅行", "景色和建筑（风光、氛围、特色）"
        
        elif any(kw in combined for kw in ['数码', '手机', '电脑', '耳机', '相机', 'iphone', 'ipad', '键盘', '鼠标']):
            return "数码产品", "产品细节（外观、设计、功能展示）"
        
        elif any(kw in combined for kw in ['好物', '物品', '文具', '家居', '用品', '工具']):
            return "好物推荐", "产品实物（摆拍、使用场景）"
        
        elif any(kw in combined for kw in ['咖啡店', '书店', '商场', '店铺', '空间', '环境']):
            return "空间环境", "店铺环境和氛围（装修、布局、细节）"
        
        elif any(kw in combined for kw in ['书', '电影', '音乐', '游戏', '剧']):
            return "文娱推荐", "相关视觉元素（封面、海报、场景）"
        
        else:
            return "生活分享", "生活场景和细节"
    
    def _extract_all_captions_from_text(self, text_content, num_images):
        """
        一次性提取所有图片的caption，确保它们不同（初次生成时使用，不分析图片）
        
        Args:
            text_content: 文案内容
            num_images: 图片数量
            
        Returns:
            caption列表，每个都是dict: {{"zh": "中文caption", "en": "English caption"}}
        """
        if num_images == 1:
            # 单张图片，直接调用单个提取函数
            caption = self._extract_caption_from_text(text_content, image_index=0)
            return [caption]
        
        try:
            # 使用AI一次性提取所有图片的caption，确保它们不同，同时生成中英文
            prompt = f"""Extract {num_images} DIFFERENT, SPECIFIC product/item names from this Xiaohongshu post text as image captions. For each image, provide BOTH Chinese and English captions.

Post text:
{text_content[:1000]}

Requirements:
1. **PRIORITY: Specific Product/Item Name**: Extract CONCRETE products, foods, drinks, or items mentioned in the post
   - ✅ Good: "拿铁咖啡" / "latte coffee", "抹茶饮品" / "matcha drink", "红色口红" / "red lipstick"
   - ❌ Bad: "咖啡店" / "coffee shop" (location, too vague), "上海甜品店" / "Shanghai dessert shop" (location, too vague)
2. **DIVERSITY IS CRITICAL**: Each caption must be DIFFERENT from others
   - If post mentions multiple items → extract different items for each image
   - If post mentions one main item → extract different aspects/details
3. **Chinese Caption**: Must be natural, idiomatic Chinese (NOT machine translation). Use appropriate Chinese terms.
4. **English Caption**: 2-6 English words, for CLIP model evaluation
5. **Concrete over Abstract**: Prefer specific items over general categories or locations
6. **Image-focused**: What would be the main subject in each image? Extract that specific item.

Output format: For each image, provide one line with format: "Image N: [Chinese] | [English]"

Example outputs for 2 images:
- Post about "抹茶茉莉拿铁" → 
  Image 1: 抹茶拿铁 | matcha latte
  Image 2: 茉莉茶 | jasmine tea
  
- Post about "拿铁咖啡和蛋糕" → 
  Image 1: 拿铁咖啡 | latte coffee
  Image 2: 蛋糕 | cake dessert

- Post about "红色口红试色" → 
  Image 1: 红色口红 | red lipstick
  Image 2: 试色效果 | lip swatch

Output (one line per image):"""

            resp = requests.post(
                f"{self.chat_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.chat_api_key}"},
                json={
                    "model": self.chat_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5,  # 稍高温度以增加多样性
                    "max_tokens": 150
                },
                timeout=30
            )
            
            if resp.status_code == 200:
                response_text = resp.json()["choices"][0]["message"]["content"].strip()
                # 解析多行输出，提取中英文
                lines = response_text.split('\n')
                captions = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析格式: "Image N: [Chinese] | [English]" 或 "1. [Chinese] | [English]"
                    if '|' in line:
                        # 分离中英文
                        parts = line.split('|', 1)
                        if len(parts) == 2:
                            # 移除前缀（Image N: 或 1. 等）
                            chinese_part = parts[0].strip()
                            english_part = parts[1].strip()
                            
                            # 清理中文部分
                            if ':' in chinese_part:
                                chinese_part = chinese_part.split(':', 1)[1].strip()
                            if chinese_part and chinese_part[0].isdigit():
                                chinese_part = chinese_part.split('.', 1)[1].strip() if '.' in chinese_part else chinese_part[1:].strip()
                            
                            # 清理英文部分
                            english_part = english_part.strip()
                            # 限制英文长度（2-6个单词）
                            words = english_part.split()
                            if len(words) > 6:
                                english_part = " ".join(words[:6])
                            
                            if chinese_part and english_part and len(words) >= 2:
                                captions.append({"zh": chinese_part, "en": english_part})
                
                # 确保数量正确
                if len(captions) == num_images:
                    return captions
                elif len(captions) > num_images:
                    return captions[:num_images]
                else:
                    # 如果提取的数量不够，用第一个caption的变体填充
                    while len(captions) < num_images:
                        if captions:
                            # 基于已有caption生成变体
                            base_caption = captions[0]
                            variant_zh = f"{base_caption['zh']}细节" if len(captions) == 1 else f"{base_caption['zh']}{len(captions)+1}"
                            variant_en = f"{base_caption['en']} detail" if len(captions) == 1 else f"{base_caption['en']} {len(captions)+1}"
                            captions.append({"zh": variant_zh, "en": variant_en})
                        else:
                            fallback_zh = self._simple_keyword_extraction(text_content) if text_content else "生活分享"
                            fallback_en = self._translate_to_english(fallback_zh)
                            captions.append({"zh": fallback_zh, "en": fallback_en})
                    return captions[:num_images]
            else:
                # 降级方案：逐个提取
                print(f"   ⚠️  批量提取caption失败，使用逐个提取方案")
                return [self._extract_caption_from_text(text_content, i) for i in range(num_images)]
                
        except Exception as e:
            print(f"   ⚠️  批量提取caption失败: {e}，使用降级方案")
            # 降级方案：逐个提取
            return [self._extract_caption_from_text(text_content, i) for i in range(num_images)]
    
    def _extract_caption_from_text(self, text_content, image_index=0):
        """
        基于文案提取单个caption（初次生成时使用，不分析图片）
        
        Args:
            text_content: 文案内容
            image_index: 图片索引（0-based）
            
        Returns:
            dict: {{"zh": "中文caption", "en": "English caption"}}
        """
        try:
            # 使用AI从文案中提取关键词，同时生成中英文
            prompt = f"""Extract a concise, SPECIFIC product/item name from this Xiaohongshu post text as image caption. Provide BOTH Chinese and English versions.

Post text:
{text_content[:800]}

Requirements:
1. **PRIORITY: Specific Product/Item Name**: Extract the CONCRETE product, food, drink, or item mentioned in the post
   - ✅ Good: "拿铁咖啡" / "latte coffee", "抹茶饮品" / "matcha drink", "红色口红" / "red lipstick"
   - ❌ Bad: "咖啡店" / "coffee shop" (location, too vague), "上海甜品店" / "Shanghai dessert shop" (location, too vague)
2. **Chinese Caption**: Must be natural, idiomatic Chinese (NOT machine translation). Use appropriate Chinese terms.
3. **English Caption**: 2-6 English words, for CLIP model evaluation
4. **Concrete over Abstract**: Prefer specific items over general categories or locations
5. **Image-focused**: What would be the main subject in the image? Extract that specific item.

Output format: "[Chinese] | [English]"

Example outputs:
- Post about "抹茶茉莉拿铁" → 抹茶拿铁 | matcha latte
- Post about "拿铁咖啡" → 拿铁咖啡 | latte coffee
- Post about "红色口红" → 红色口红 | red lipstick
- Post about "白色运动鞋" → 白色运动鞋 | white sneakers

Output:"""

            resp = requests.post(
                f"{self.chat_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.chat_api_key}"},
                json={
                    "model": self.chat_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 80
                },
                timeout=30
            )
            
            if resp.status_code == 200:
                response_text = resp.json()["choices"][0]["message"]["content"].strip()
                # 解析中英文
                if '|' in response_text:
                    parts = response_text.split('|', 1)
                    chinese = parts[0].strip()
                    english = parts[1].strip()
                    # 限制英文长度（2-6个单词）
                    words = english.split()
                    if len(words) > 6:
                        english = " ".join(words[:6])
                    if chinese and english and len(words) >= 2:
                        return {"zh": chinese, "en": english}
                
                # 如果解析失败，降级方案
                fallback_zh = self._simple_keyword_extraction(text_content) if text_content else "生活分享"
                fallback_en = self._translate_to_english(fallback_zh)
                return {"zh": fallback_zh, "en": fallback_en}
            else:
                fallback_zh = self._simple_keyword_extraction(text_content) if text_content else "生活分享"
                fallback_en = self._translate_to_english(fallback_zh)
                return {"zh": fallback_zh, "en": fallback_en}
                
        except Exception as e:
            print(f"   ⚠️  文案提取caption失败: {e}，使用降级方案")
            fallback_zh = self._simple_keyword_extraction(text_content) if text_content else "生活分享"
            fallback_en = self._translate_to_english(fallback_zh)
            return {"zh": fallback_zh, "en": fallback_en}
    
    def _extract_keywords_for_caption(self, image_path, text_content=None):
        """
        基于图片内容提取核心主体作为caption（使用vision model分析图片）
        
        Args:
            image_path: 图片文件路径
            text_content: 文案内容（可选，作为上下文参考）
            
        Returns:
            dict: {{"zh": "中文caption", "en": "English caption"}}
        """
        if not image_path or not os.path.exists(image_path):
            print(f"   ⚠️  图片不存在，使用降级方案")
            fallback_zh = self._simple_keyword_extraction(text_content) if text_content else "生活分享"
            fallback_en = self._translate_to_english(fallback_zh)
            return {"zh": fallback_zh, "en": fallback_en}
        
        try:
            # 读取图片并编码为base64
            with open(image_path, 'rb') as f:
                img_data = f.read()
            
            # 判断图片格式
            ext = Path(image_path).suffix.lower()
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.webp': 'image/webp',
                '.gif': 'image/gif'
            }.get(ext, 'image/jpeg')
            
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # 构建vision prompt（专注于提取核心主体，同时生成中英文）
            context_hint = ""
            if text_content:
                context_hint = f"""
**文案上下文（仅供参考）**：
{text_content[:200]}
"""
            
            vision_prompt = f"""Analyze this image and extract a concise core subject as caption. Provide BOTH Chinese and English versions.

{context_hint}

**Requirements**:
1. **Core Subject**: Extract only the most prominent visual element in the image
2. **Chinese Caption**: Must be natural, idiomatic Chinese (NOT machine translation). Use appropriate Chinese terms.
3. **English Caption**: 2-6 English words, for CLIP model evaluation
4. **Specific but not verbose**:
   - ✅ Good: "拿铁咖啡" / "latte coffee", "抹茶饮品" / "matcha drink"
   - ❌ Bad: "食物" / "food" (too vague)
5. **Based on image content**: Must accurately reflect what's actually shown in the image, don't guess from text

**Output format**: "[Chinese] | [English]"

Example outputs:
- Image shows coffee and cup → 拿铁咖啡 | latte coffee
- Image shows shop interior → 咖啡店 | coffee shop
- Image shows product display → 产品展示 | product review
- Image shows scenery/building → 旅行风景 | travel vlog

Output:"""

            # 使用vision model（如果有配置）或chat model
            vision_model = os.getenv("VISION_MODEL", "claude-sonnet-4-5-20250929")
            search_api_key = os.getenv("SEARCH_API_KEY", self.chat_api_key)
            search_base_url = os.getenv("SEARCH_BASE_URL", self.chat_base_url)
            
            response = requests.post(
                f"{search_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {search_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": vision_prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{img_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 50
                },
                timeout=30
            )
            
            if response.status_code == 200:
                response_text = response.json()["choices"][0]["message"]["content"].strip()
                # 解析中英文
                if '|' in response_text:
                    parts = response_text.split('|', 1)
                    chinese = parts[0].strip()
                    english = parts[1].strip()
                    # 清理可能的格式
                    chinese = chinese.replace("关键词：", "").replace("关键词:", "").strip()
                    english = english.replace("Keywords:", "").replace("Keywords：", "").strip()
                    # 限制英文长度（2-6个单词）
                    words = english.split()
                    if len(words) > 6:
                        english = " ".join(words[:6])
                    if chinese and english and len(words) >= 2:
                        return {"zh": chinese, "en": english}
                
                # 如果解析失败，降级方案
                fallback_zh = self._simple_keyword_extraction(text_content) if text_content else "生活分享"
                fallback_en = self._translate_to_english(fallback_zh)
                return {"zh": fallback_zh, "en": fallback_en}
            else:
                print(f"   ⚠️  Vision API错误 {response.status_code}，使用降级方案")
                fallback_zh = self._simple_keyword_extraction(text_content) if text_content else "生活分享"
                fallback_en = self._translate_to_english(fallback_zh)
                return {"zh": fallback_zh, "en": fallback_en}
                
        except Exception as e:
            print(f"   ⚠️  图片分析失败: {e}，使用降级方案")
            fallback_zh = self._simple_keyword_extraction(text_content) if text_content else "生活分享"
            fallback_en = self._translate_to_english(fallback_zh)
            return {"zh": fallback_zh, "en": fallback_en}
    
    def _simple_keyword_extraction(self, text_content):
        """简单的关键词提取（降级方案，返回中文）"""
        # 提取常见关键词
        keywords_map = {
            "咖啡": ["咖啡", "拿铁", "美式", "卡布", "手冲"],
            "美食": ["美食", "餐厅", "火锅", "烧烤", "料理"],
            "旅行": ["旅行", "旅游", "景点", "风景"],
            "好物": ["好物", "分享", "推荐", "测评"],
            "生活": ["日常", "生活", "周末", "日记"]
        }
        
        text_lower = text_content.lower() if text_content else ""
        for category, keywords in keywords_map.items():
            for kw in keywords:
                if kw in text_lower:
                    return category
        
        return "生活分享"
    
    def _translate_to_english(self, chinese_text):
        """将中文关键词翻译成英文（用于CLIP评估）"""
        translation_map = {
            "咖啡": "coffee drink",
            "美食": "food",
            "旅行": "travel",
            "好物": "product review",
            "生活分享": "lifestyle",
            "生活": "lifestyle",
            "探店": "cafe visit",
            "咖啡店": "coffee shop",
            "抹茶饮品": "matcha drink",
            "旅行vlog": "travel vlog"
        }
        
        # 直接匹配
        if chinese_text in translation_map:
            return translation_map[chinese_text]
        
        # 尝试部分匹配
        for key, value in translation_map.items():
            if key in chinese_text:
                return value
        
        # 如果找不到，使用简单翻译
        try:
            # 使用AI快速翻译
            prompt = f"Translate this Chinese keyword to English (2-6 words, for image caption): {chinese_text}\n\nOutput ONLY the English translation, no other text:"
            
            resp = requests.post(
                f"{self.chat_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.chat_api_key}"},
                json={
                    "model": self.chat_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 20
                },
                timeout=10
            )
            
            if resp.status_code == 200:
                translation = resp.json()["choices"][0]["message"]["content"].strip()
                # 清理格式
                translation = translation.replace("Translation:", "").replace("翻译:", "").strip()
                return translation
        except:
            pass
        
        # 最终降级：返回通用英文
        return "lifestyle"
    
    def _extract_generalized_theme(self, text_content):
        """
        从完整文案中提取极度泛化的主题
        只保留最宽泛的主题类别，移除所有具体细节，让图片生成时完全看不到具体视觉描述
        """
        import re
        
        # 第一步：移除所有具体的视觉描述词汇
        text = text_content
        
        # 移除所有颜色描述
        text = re.sub(r'\b(白色|黑色|红色|蓝色|绿色|黄色|粉色|紫色|棕色|灰色|米色|咖啡色|深色|浅色|亮色|暗色|金色|银色|透明|半透明)\s*(的|)?', '', text)
        
        # 移除所有具体物品细节
        detail_patterns = [
            r'心形[的]?拉花', r'拉花图案', r'图案', r'花纹', r'纹理',
            r'陶瓷杯', r'玻璃杯', r'马克杯', r'咖啡杯', r'杯子',
            r'木桌', r'木制', r'木质', r'桌子', r'桌面',
            r'特写', r'细节', r'质感', r'材质',
            r'摆盘', r'装饰', r'点缀', r'搭配',
            r'拿铁', r'美式', r'卡布奇诺', r'手冲',  # 具体咖啡类型
            r'iPhone', r'MacBook', r'AirPods',  # 具体产品型号
            r'朝阳', r'三里屯', r'太古里',  # 具体地点
            r'人均\d+', r'\d+元', r'价格', r'折扣',  # 价格信息
        ]
        for pattern in detail_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 第二步：只提取最宽泛的主题类别（极度简化）
        # 只保留最核心的主题词，不包含任何具体描述
        theme_keywords = []
        text_lower = text.lower()
        
        # 使用更宽泛的分类
        if any(kw in text_lower for kw in ['咖啡', '咖啡店', '饮品', '奶茶', '茶']):
            theme_keywords.append('饮品场景')  # 不说是咖啡，只说是饮品
        elif any(kw in text_lower for kw in ['美食', '餐厅', '火锅', '烧烤', '食物', '菜', '料理']):
            theme_keywords.append('餐饮场景')  # 不说是火锅，只说是餐饮
        elif any(kw in text_lower for kw in ['旅行', '旅游', '景点', '风景', '建筑']):
            theme_keywords.append('旅行场景')
        elif any(kw in text_lower for kw in ['数码', '手机', '电脑', '电子', '科技']):
            theme_keywords.append('科技产品')  # 不说是iPhone，只说是科技产品
        elif any(kw in text_lower for kw in ['好物', '物品', '产品', '商品']):
            theme_keywords.append('产品展示')
        elif any(kw in text_lower for kw in ['书', '电影', '音乐', '游戏', '剧', '娱乐']):
            theme_keywords.append('文娱内容')
        else:
            theme_keywords.append('生活场景')  # 最宽泛的分类
        
        # 第三步：移除情感词，只保留最宽泛的主题
        # 不包含情感描述，让图片生成更随机
        
        # 构建极度泛化的描述（只包含最宽泛的主题，不包含任何具体信息）
        generalized = theme_keywords[0]  # 只保留主题，不包含情感
        
        return generalized

    def generate_images(self, user_profile, text_content, output_dir, tags=None, image_captions=None):
        """生成配图，智能判断是否需要人物，保持系列图片一致性
        
        Args:
            user_profile: 用户画像
            text_content: 文案内容（用于判断图片数量和生成相关图片）
            output_dir: 输出目录
            tags: 笔记标签列表
            image_captions: 预提取的caption列表（可选，生成图片时不参考，仅用于后续HTML生成）
        """
        word_count = len(text_content.strip())
        num_images = 1 if word_count <= 300 else (2 if word_count <= 800 else 3)
        print(f"文案字数: {word_count}, 需要生成 {num_images} 张图片")

        # 分析文案内容，确定图片主题和聚焦对象
        theme, focus_desc = self._get_focus_subject(text_content, user_profile)
        must_people = self._must_include_people(text_content, user_profile)
        must_pets = self._must_include_pets(text_content, user_profile)
        
        # 构建聚焦指引
        if must_people:
            focus_guidance = f"""
📍 主题：{theme}
⚠️ **必须包含人物**（如穿搭展示、健身、自拍等场景）
聚焦对象：人物展示，配合 {focus_desc.lower()}
"""
            consistency_mode = "人物"
        elif must_pets:
            focus_guidance = f"""
📍 主题：{theme}
⚠️ **必须包含宠物**
聚焦对象：宠物特写和日常
"""
            consistency_mode = "宠物"
        else:
            focus_guidance = f"""
📍 主题：{theme}
聚焦对象：{focus_desc}
"""
            consistency_mode = "通用"
        
        style_reference = f"\n参考风格：小红书真实图片风格，自然、生活化、有质感。"
        
        # 如果传入了预提取的captions，说明是初次生成，只使用tags，不传入文案
        # 如果没有传入，则使用文案内容（降级方案）
        use_pre_extracted_captions = image_captions is not None and len(image_captions) >= num_images
        
        if use_pre_extracted_captions:
            print(f"🎨 生成策略：初次生成 - 仅使用tags，不传入文案上下文")
        else:
            print(f"🎨 图片主题：{theme}")
            print(f"📸 生成策略：基于文案内容生成相关图片")

        image_paths = []
        max_retries = 5  # 增加重试次数（从3增加到5）
        reference_image = None  # 用于保持一致性

        for i in range(num_images):
            image_path = os.path.join(output_dir, f"personalized_post_{i+1}.png")
            success = False
            
            for attempt in range(1, max_retries + 1):
                if attempt > 1:
                    print(f"   🔄 第 {i+1} 张图片重试 {attempt}/{max_retries}...")
                
                # 🎯 构建提示词：初次生成时只使用tags，不传入文案
                if use_pre_extracted_captions:
                    # 初次生成：只使用tags，不传入文案上下文
                    tags_section = ""
                    if tags:
                        tags_str = ", ".join(tags)
                        tags_section = f"""
**笔记标签（唯一参考）**：
{tags_str}

⚠️ **重要**：只根据标签生成图片，不要参考其他内容。标签是：{tags_str}
"""
                    
                    if i == 0:
                        prompt_text = f"""
生成一张小红书风格的图片（第 {i+1}/{num_images} 张）：

{tags_section}

{style_reference}

💡 **生成指导**：
- 根据标签主题生成相关的图片
- 保持小红书真实图片风格，自然、生活化、有质感

要求：
- 真实自然的社交媒体照片风格
- 最好不要出现文字，千万不要出现中文或什么奇怪的字符！实在需要文字请使用英文，并且尽量放在图片的角落，不要影响图片主体
- 符合小红书图片美学
- 生活化、有质感、有故事感
- 构图清晰，主体突出
- 色彩自然，光线柔和
- 如果画面中出现了人物或动物，请确保其特征明确，便于后续图片保持一致
- 后续图片关键人物或动物保持一致，但是突出的事物需要改变，避免图片间过度相似
"""
                    else:
                        prompt_text = f"""
生成一张小红书风格的图片（第 {i+1}/{num_images} 张，参考前一张图片）：

{tags_section}

{style_reference}

⚠️ **风格一致性要求**
- 与参考图片保持整体风格和构图思路一致
- 如果参考图片中有人物或动物，保持其外貌特征完全一致
- 可以改变：场景、角度、动作、背景、拍摄距离
- 突出的事物需要改变，避免图片间过度相似
- 保持色调、氛围和拍摄风格的连贯性

要求：
- 真实自然的社交媒体照片风格
- **绝对禁止在图片上生成任何文字，特别是中文文字！** 图片必须是纯视觉内容，不能有任何文字叠加
- 生活化、有质感、有故事感
"""
                else:
                    # 降级方案：使用文案内容（如果没有预提取的captions）
                    if i == 0:
                        prompt_text = f"""
生成一张小红书风格的图片（第 {i+1}/{num_images} 张）：

**文案内容（参考）**：
{text_content[:500]}

{style_reference}

{focus_guidance}

💡 **生成指导**：
- 根据文案内容生成相关的图片
- 图片应该展示文案中提到的场景、物品或主题
- 保持小红书真实图片风格，自然、生活化、有质感

要求：
- 真实自然的社交媒体照片风格
- 最好不要出现文字，千万不要出现中文或什么奇怪的字符！实在需要文字请使用英文，并且尽量放在图片的角落，不要影响图片主体
- 符合小红书图片美学
- 生活化、有质感、有故事感
- 构图清晰，主体突出
- 色彩自然，光线柔和
- 如果画面中出现了人物或动物，请确保其特征明确，便于后续图片保持一致
- 后续图片关键人物或动物保持一致，但是突出的事物需要改变，避免图片间过度相似
"""
                    else:
                        prompt_text = f"""
生成一张小红书风格的图片（第 {i+1}/{num_images} 张，参考前一张图片）：

**文案内容（参考）**：
{text_content[:500]}

{style_reference}

⚠️ **风格一致性要求**
- 与参考图片保持整体风格和构图思路一致
- 如果参考图片中有人物或动物，保持其外貌特征完全一致
- 可以改变：场景、角度、动作、背景、拍摄距离
- 突出的事物需要改变，避免图片间过度相似
- 保持色调、氛围和拍摄风格的连贯性

{focus_guidance}

💡 **生成指导**：
- 根据文案内容生成相关的图片
- 图片应该展示文案中提到的场景、物品或主题
- 保持与前一张图片的风格一致性

要求：
- 真实自然的社交媒体照片风格
- **绝对禁止在图片上生成任何文字，特别是中文文字！** 图片必须是纯视觉内容，不能有任何文字叠加
- 生活化、有质感、有故事感
"""
                
                try:
                    # 构建请求内容
                    parts = [{"text": prompt_text}]
                    
                    # 如果有参考图片且不是第一张，添加参考
                    if reference_image and i > 0:
                        parts.insert(0, {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": reference_image
                            }
                        })
                    
                    resp = requests.post(
                        f"{self.generate_base_url}/models/gemini-2.5-flash-image-preview:generateContent",
                        headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.generate_api_key}"},
                        json={
                            "contents": [{"parts": parts}],
                            "generationConfig": {"temperature": 0.9}  # 正常temperature，生成自然相关的图片
                        },
                        timeout=60
                    )
                    
                    if resp.status_code == 200:
                        result_parts = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
                        for part in result_parts:
                            if "inlineData" in part:
                                # 保存图片
                                image_data = base64.b64decode(part["inlineData"]["data"])
                                with open(image_path, "wb") as f:
                                    f.write(image_data)
                                
                                image_paths.append(image_path)
                                
                                # 🎯 如果已有预提取的caption，不重新提取
                                # 如果没有，则分析图片提取caption（降级方案）
                                if not use_pre_extracted_captions:
                                    caption_keyword = self._extract_keywords_for_caption(image_path, text_content=text_content)
                                    if image_captions is None:
                                        image_captions = []
                                    image_captions.append(caption_keyword)
                                    if isinstance(caption_keyword, dict):
                                        print(f"   📝 分析图片提取caption: {caption_keyword.get('zh', '')} ({caption_keyword.get('en', '')})")
                                    else:
                                        print(f"   📝 分析图片提取caption: {caption_keyword}")
                                
                                # 保存为下一张的参考
                                reference_image = part["inlineData"]["data"]
                                
                                success = True
                                if i == 0:
                                    print(f"✅ 第 {i+1} 张图片已保存（基准图片）")
                                else:
                                    if consistency_mode in ["人物", "宠物"]:
                                        print(f"✅ 第 {i+1} 张图片已保存（保持{consistency_mode}一致）")
                                    else:
                                        print(f"✅ 第 {i+1} 张图片已保存（保持风格一致）")
                                break
                    
                    if success:
                        break
                    else:
                        # 请求成功但没有生成图片，打印详细信息
                        if resp.status_code != 200:
                            print(f"   ⚠️  API返回错误: {resp.status_code}")
                            if resp.text:
                                error_msg = resp.text[:200]
                                print(f"       错误信息: {error_msg}")
                    
                except Exception as e:
                    print(f"   ⚠️  图片生成异常 (尝试 {attempt}/{max_retries}): {str(e)[:100]}")
                    import time
                    if attempt < max_retries:
                        time.sleep(2)  # 等待2秒后重试
            
            if not success:
                print(f"❌ 第 {i+1} 张图片生成失败（已重试{max_retries}次）")
                if i == 0:
                    print("⚠️ 基准图片生成失败")
                    print("💡 提示：将在Reflection阶段尝试重新生成图片")
                    break  # 不再尝试后续图片
        
        # 如果使用预提取的captions，直接返回它们
        # 如果没有预提取的captions，确保image_paths和image_captions长度一致
        if not use_pre_extracted_captions:
            if image_captions is None:
                image_captions = []
            while len(image_captions) < len(image_paths):
                image_captions.append(None)
        
        # 返回图片路径（captions已在外部预提取或已生成）
        return image_paths

    def _render_markdown(self, text):
        """
        简单的 Markdown 转 HTML 转换器
        解决网页上显示 **text** 符号的问题，将其转换为加粗样式
        """
        # 1. 处理加粗 **text** -> <strong>text</strong>
        # 使用 ? 非贪婪匹配，防止跨行匹配过多
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #333; font-weight: 700;">\1</strong>', text)
        
        # 2. 处理简单的标题 (以防万一 AI 输出了标题)
        # ### Title -> <h3>Title</h3>
        text = re.sub(r'^###\s+(.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        
        # 3. 处理列表项 (简单的 - item)
        text = re.sub(r'^\-\s+(.*?)$', r'• \1', text, flags=re.MULTILINE)

        return text

    def generate_html_post(self, text_content, image_paths, links, tags, output_path="post.html", image_captions=None):
        """生成HTML帖子，包含 Markdown 渲染逻辑
        
        Args:
            text_content: 文本内容
            image_paths: 图片路径列表
            links: 链接列表
            tags: 标签列表
            output_path: 输出路径
            image_captions: 图片caption列表（可选）
        """
        
        # 按双换行符分割段落
        raw_paragraphs = [p for p in text_content.split('\n\n') if p.strip()]
        
        # --- 核心修改：渲染每一段的 Markdown ---
        processed_paragraphs = []
        for p in raw_paragraphs:
            rendered_p = self._render_markdown(p)
            processed_paragraphs.append(rendered_p)
            
        paragraphs = processed_paragraphs
        # ------------------------------------

        insertions = []
        
        # 插入图片（带caption）
        for i, img_path in enumerate(image_paths):
            caption = image_captions[i] if image_captions and i < len(image_captions) else None
            insertions.append({"type": "image", "content": img_path, "index": i, "caption": caption})
        # 插入链接
        for link in links:
            insertions.append({"type": "link", "content": link})
            
        html_parts = []
        num_paras = len(paragraphs)
        
        if num_paras == 0:
            html_parts.append(text_content)
        else:
            num_inserts = len(insertions)
            if num_inserts > 0:
                step = max(1, num_paras // (num_inserts + 1))
                current_insert_idx = 0
                for i, para in enumerate(paragraphs):
                    # 判断是否已经是标题标签（如果是h3就不加p标签了）
                    if para.startswith('<h3'):
                         html_parts.append(para)
                    else:
                         html_parts.append(f"<p>{para}</p>")
                         
                    if current_insert_idx < num_inserts:
                        if (i + 1) % step == 0 or i == num_paras - 1:
                            item = insertions[current_insert_idx]
                            if item["type"] == "image":
                                caption = item.get("caption")
                                html_parts.append(self._create_image_tag(item["content"], item["index"], caption=caption))
                            elif item["type"] == "link":
                                html_parts.append(self._create_link_tag(item["content"]))
                            current_insert_idx += 1
                            if i == num_paras - 1:
                                while current_insert_idx < num_inserts:
                                    item = insertions[current_insert_idx]
                                    if item["type"] == "image":
                                        caption = item.get("caption")
                                        html_parts.append(self._create_image_tag(item["content"], item["index"], caption=caption))
                                    elif item["type"] == "link":
                                        html_parts.append(self._create_link_tag(item["content"]))
                                    current_insert_idx += 1
            else:
                for para in paragraphs:
                    if para.startswith('<h3'):
                        html_parts.append(para)
                    else:
                        html_parts.append(f"<p>{para}</p>")

        html_content = "\n".join(html_parts)
                
        html_template = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>个性化社交媒体帖子</title>
            <style>
                body {{ font-family: 'Helvetica Neue', Helvetica, 'Microsoft YaHei', sans-serif; line-height: 1.75; max-width: 800px; margin: 0 auto; padding: 15px; background: #f5f5f5; color: #333; }}
                .post-container {{ background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin: 10px 0; }}
                
                /* 正文样式优化 */
                .post-content {{ font-size: 17px; color: #2c3e50; letter-spacing: 0.02em; }}
                .post-content p {{ margin: 1em 0; text-align: justify; }}
                .post-content strong {{ color: #000; font-weight: 700; background: linear-gradient(to bottom, transparent 60%, #fffbe6 60%); }} /* 模拟高亮笔效果 */
                .post-content h3 {{ font-size: 1.2em; margin-top: 1.5em; margin-bottom: 0.5em; color: #1a1a1a; }}
                
                .post-image {{ margin: 20px -25px; width: calc(100% + 50px); text-align: center; }}
                .post-image img {{ width: 100%; display: block; }}
                .image-caption {{ color: #999; font-size: 13px; margin-top: 8px; font-style: italic; padding: 0 25px; }}

                /* 标签样式 */
                .post-tags {{ 
                    margin: 15px 0 20px 0; 
                    display: flex; 
                    flex-wrap: wrap; 
                    gap: 8px;
                }}
                .tag {{ 
                    display: inline-block; 
                    padding: 5px 12px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    font-size: 13px; 
                    border-radius: 15px; 
                    font-weight: 500;
                    box-shadow: 0 2px 4px rgba(102, 126, 234, 0.2);
                }}

                /* 链接卡片样式 (保持不变) */
                .link-card {{
                    display: flex;
                    align-items: center;
                    background: #fcfcfc;
                    border: 1px solid #eee;
                    padding: 12px 15px;
                    margin: 25px 0;
                    text-decoration: none;
                    border-radius: 8px;
                    transition: all 0.2s;
                }}
                .link-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
                .link-card.platform-b站 {{ border-left: 5px solid #23ade5; }}
                .link-card.platform-小红书 {{ border-left: 5px solid #ff2442; }}
                .link-card.platform-知乎 {{ border-left: 5px solid #0084ff; }}
                .link-card.platform-抖音 {{ border-left: 5px solid #1c1e21; }}
                .link-card.platform-微博 {{ border-left: 5px solid #ea5d5c; }}
                
                .link-info {{ flex: 1; }}
                .link-platform-tag {{ 
                    font-size: 12px; font-weight: bold; margin-bottom: 4px; display: inline-block; padding: 2px 6px; border-radius: 4px; color: white;
                }}
                .tag-b站 {{ background: #23ade5; }}
                .tag-小红书 {{ background: #ff2442; }}
                .tag-知乎 {{ background: #0084ff; }}
                .tag-抖音 {{ background: #000; }}
                .tag-微博 {{ background: #ea5d5c; }}
                
                .link-title {{ font-weight: bold; color: #333; font-size: 15px; margin-top: 2px; }}
                .link-action {{ color: #999; font-size: 12px; margin-top: 4px; }}
                .link-icon {{ font-size: 24px; margin-right: 15px; }}

                .post-header {{ display: flex; align-items: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid #f0f0f0; }} 
                .avatar {{ width: 48px; height: 48px; border-radius: 50%; background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%); margin-right: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }} 
                .user-info h3 {{ margin: 0; font-size: 18px; font-weight: 600; }} 
                .post-time {{ color: #999; font-size: 13px; margin-top: 4px; }} 

                @media (max-width: 600px) {{ 
                    .post-container {{ padding: 15px; }} 
                    .post-image {{ margin: 15px -15px; width: calc(100% + 30px); }}
                }}
            </style>
        </head>
        <body>
            <div class="post-container">
                <div class="post-header">
                    <div class="avatar"></div>
                    <div class="user-info">
                        <h3>AI 创意助理</h3>
                        <div class="post-time">{datetime.now().strftime('%Y年%m月%d日 %H:%M')}</div>
                    </div>
                </div>
                <div class="post-tags">
                    {"".join([f'<span class="tag"># {tag}</span>' for tag in tags])}
                </div>
                <div class="post-content">{html_content}</div>
                <div style="margin-top:30px; border-top:1px solid #eee; padding-top:15px; color:#ccc; font-size:12px; text-align:center;">
                    Generated by AI • {len(image_paths)} Images
                </div>
            </div>
        </body>
        </html>
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        return output_path

    def _create_image_tag(self, image_path, index, caption=None):
        """
        创建图片标签，使用提取的caption
        
        Args:
            image_path: 图片路径
            index: 图片索引（0-based）
            caption: 图片caption，可以是dict {{"zh": "中文", "en": "English"}} 或字符串（向后兼容）
        """
        if caption:
            # 处理新的dict格式或旧的字符串格式（向后兼容）
            if isinstance(caption, dict):
                caption_zh = caption.get("zh", "")
                caption_en = caption.get("en", "")
                # 显示中文caption，用data属性存储英文caption（用于CLIP计算）
                caption_text = f"图{index + 1}: {caption_zh}"
                if caption_en:
                    return f'<div class="post-image"><img src="{os.path.basename(image_path)}"><div class="image-caption" data-caption-en="{caption_en}">{caption_text}</div></div>'
                else:
                    return f'<div class="post-image"><img src="{os.path.basename(image_path)}"><div class="image-caption">{caption_text}</div></div>'
            else:
                # 向后兼容：如果是字符串，假设是英文
                caption_text = f"图{index + 1}: {caption}"
                return f'<div class="post-image"><img src="{os.path.basename(image_path)}"><div class="image-caption" data-caption-en="{caption}">{caption_text}</div></div>'
        else:
            # 降级：使用默认格式
            caption_text = f"图 {index + 1}"
            return f'<div class="post-image"><img src="{os.path.basename(image_path)}"><div class="image-caption">{caption_text}</div></div>'

    def _create_link_tag(self, link_data):
        title = link_data.get('title', '相关内容')
        platform = link_data.get('platform', '网页').strip()
        url = link_data.get('url', '#')
        
        css_class = "platform-other"
        tag_class = "tag-other"
        icon = "🔗"
        
        p = platform.lower()
        if "b站" in p or "bilibili" in p:
            css_class = "platform-b站"
            tag_class = "tag-b站"
            icon = "📺"
        elif "小红书" in p:
            css_class = "platform-小红书"
            tag_class = "tag-小红书"
            icon = "📕"
        elif "知乎" in p:
            css_class = "platform-知乎"
            tag_class = "tag-知乎"
            icon = "❓"
        elif "抖音" in p:
            css_class = "platform-抖音"
            tag_class = "tag-抖音"
            icon = "🎵"
        elif "微博" in p:
            css_class = "platform-微博"
            tag_class = "tag-微博"
            icon = "👁️"
            
        return f'''
        <a href="{url}" class="link-card {css_class}" target="_blank">
            <div class="link-icon">{icon}</div>
            <div class="link-info">
                <span class="link-platform-tag {tag_class}">{platform}</span>
                <div class="link-title">{title}</div>
                <div class="link-action">点击去 {platform} 查看详情 &gt;</div>
            </div>
        </a>
        '''

    def _extract_captions_from_html(self, html_path):
        """
        从HTML中提取现有的captions
        
        Returns:
            caption列表，如果提取失败则返回空列表
        """
        try:
            from bs4 import BeautifulSoup
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            content_div = soup.find('div', class_='post-content')
            
            if not content_div:
                return []
            
            captions = []
            image_divs = content_div.find_all('div', class_='post-image')
            
            for img_div in image_divs:
                caption_div = img_div.find('div', class_='image-caption')
                if caption_div:
                    # 优先从data-caption-en读取英文，从文本读取中文
                    caption_en = caption_div.get('data-caption-en', '')
                    caption_text = caption_div.get_text(strip=True)
                    # 移除"图X:"前缀（如果有）
                    if caption_text.startswith("图") and ":" in caption_text:
                        caption_text = caption_text.split(":", 1)[1].strip()
                    
                    # 返回dict格式（包含中英文）
                    if caption_en:
                        captions.append({"zh": caption_text, "en": caption_en})
                    else:
                        # 向后兼容：如果没有英文，假设文本是英文
                        captions.append({"zh": caption_text, "en": caption_text})
                else:
                    captions.append(None)
            
            return captions
        except Exception as e:
            print(f"   ⚠️  提取caption失败: {e}")
            return []
    
    def _regenerate_images_for_reflection(self, html_path, text_content, user_profile, output_dir, iteration, tags=None):
        """
        在Reflection过程中重新生成图片
        
        Args:
            html_path: 当前HTML文件路径
            text_content: 当前文本内容
            user_profile: 用户画像
            output_dir: 输出目录
            iteration: 当前迭代次数
            tags: 标签列表（用于第一次反思时参考）
            
        Returns:
            新图片路径列表，如果失败则返回None
        """
        try:
            print(f"\n🎨 重新生成图片（第{iteration+1}次Reflection - 图片重建策略）...")
            
            # 1. 解析当前HTML获取图片信息
            from bs4 import BeautifulSoup
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            content_div = soup.find('div', class_='post-content')
            
            if not content_div:
                print("   ⚠️  无法找到post-content div")
                return None
            
            # 获取当前图片数量
            current_images = content_div.find_all('div', class_='post-image')
            num_images = len(current_images)
            
            if num_images == 0:
                print("   ⚠️  没有图片需要重新生成")
                return None
            
            print(f"   📸 需要重新生成 {num_images} 张图片")
            print(f"   💡 策略：基于当前文本内容，重新生成与文字更匹配的图片")
            
            # 2. 读取旧图片作为参考（风格和构图）
            old_image_paths = []
            for img_div in current_images:
                img_tag = img_div.find('img')
                if img_tag and img_tag.get('src'):
                    old_img_path = os.path.join(os.path.dirname(html_path), img_tag['src'])
                    if os.path.exists(old_img_path):
                        old_image_paths.append(old_img_path)
            
            # 3. 生成新图片（使用改进的prompt，强调与文本的语义一致性）
            theme, focus_desc = self._get_focus_subject(text_content, user_profile)
            must_people = self._must_include_people(text_content, user_profile)
            must_pets = self._must_include_pets(text_content, user_profile)
            
            # 构建增强的聚焦指引（强调文本匹配）
            if must_people:
                focus_guidance = f"""
📍 主题：{theme}
⚠️ **必须包含人物**（如穿搭展示、健身、自拍等场景）
聚焦对象：人物展示，配合 {focus_desc.lower()}
🎯 **关键要求**：图片内容必须与文案强相关，展示文案中提到的具体场景、动作、物品
"""
            elif must_pets:
                focus_guidance = f"""
📍 主题：{theme}
⚠️ **必须包含宠物**
聚焦对象：宠物特写和日常
🎯 **关键要求**：图片内容必须与文案强相关，展示文案中提到的具体场景、动作
"""
            else:
                focus_guidance = f"""
📍 主题：{theme}
聚焦对象：{focus_desc}
🎯 **关键要求**：图片必须精确展示文案中描述的内容（产品、场景、细节、颜色、氛围）
"""
            
            new_image_paths = []
            reference_image = None
            
            # 第二轮反思（iteration == 1）时不使用旧图片参考，只看文本上下文
            # 这样可以完全基于文本内容重新生成，不受旧图片风格影响
            if iteration == 1:
                print(f"   📝 第二轮反思：不使用旧图片参考，完全基于文本上下文重新生成")

            
            for i in range(num_images):
                image_path = os.path.join(output_dir, f"personalized_post_{i+1}_v{iteration+1}.png")
                success = False
                max_retries = 3
                
                # 第一次反思（iteration == 0）：主要参考上下文和标签
                if iteration == 0:
                    tags_section = ""
                    if tags:
                        tags_str = ", ".join(tags)
                        tags_section = f"""
**Tags (Primary Reference):**
{tags_str}

**Text Context (Secondary Reference):**
{text_content[:500]}
"""
                    else:
                        tags_section = f"""
**Text Context:**
{text_content[:500]}
"""
                    
                    prompt_text = f"""
Generate image {i+1}/{num_images} for a Xiaohongshu post (1st Reflection - regenerate based on context and tags).

{tags_section}

**User Profile:**
{user_profile[:300]}

{focus_guidance}

**CRITICAL Requirements (1st Reflection - regenerate images based on context and tags)**:
1. **Primary focus on tags**: The image should strongly reflect the core themes and keywords from the tags
2. **Text context alignment**: Incorporate key visual elements mentioned in the text context
3. **Natural Xiaohongshu photo style**: Realistic, life-like, social media aesthetic
4. **Clear composition**: Main subject should be prominent and clearly visible
5. **Soft lighting and natural colors**: Maintain authentic, appealing visual quality
6. **ABSOLUTELY NO TEXT OVERLAYS**: Strictly prohibit any text on the image, especially Chinese characters! The image must be pure visual content without any text overlays
7. **If characters/pets appear**: Keep their features consistent across all images

**Style Reference**: Natural Xiaohongshu lifestyle photography with high quality and storytelling

**REMEMBER**: This is the 1st reflection - regenerate images primarily based on tags and context to improve image-text matching.
"""
                # 构建强调文本匹配的prompt（使用完整文案，确保图文高度关联）
                elif i == 0:
                    prompt_text = f"""
Generate image {i+1}/{num_images} for a Xiaohongshu post with MAXIMUM semantic consistency with the text.

**Text Content (MUST match this exactly - extract ALL visual elements):**
{text_content}

**User Profile:**
{user_profile[:300]}

{focus_guidance}

**CRITICAL Requirements (for optimal CLIP score - this is REFLECTION optimization)**:
1. **SEMANTIC ALIGNMENT IS PARAMOUNT**: The image MUST visually represent EVERY key concept, object, scene, color, and detail mentioned in the text
2. **Extract and visualize ALL keywords from text**:
   - Objects: Extract all concrete nouns (e.g., "latte", "wooden table", "white cup", "heart pattern")
   - Colors: Extract all color descriptions (e.g., "white", "brown", "golden")
   - Actions: Extract all action verbs (e.g., "drinking", "sitting", "holding")
   - Scenes: Extract all scene descriptions (e.g., "coffee shop", "outdoor", "morning light")
   - Details: Extract all specific details (e.g., "heart latte art", "ceramic cup", "wooden texture")
3. **Visualize text descriptions literally**: If text says "heart-shaped latte art", show exactly that. If text says "white ceramic cup", show exactly that.
4. **Natural Xiaohongshu photo style**: Realistic, life-like, social media aesthetic
5. **Clear composition**: Main subject should be prominent and clearly visible
6. **Soft lighting and natural colors**: Maintain authentic, appealing visual quality
7. **ABSOLUTELY NO TEXT OVERLAYS**: Strictly prohibit any text on the image, especially Chinese characters! The image must be pure visual content without any text overlays
8. **If characters/pets appear**: Keep their features consistent across all images

**Style Reference**: Natural Xiaohongshu lifestyle photography with high quality and storytelling

**REMEMBER**: This is a REFLECTION optimization - the goal is to MAXIMIZE image-text semantic matching for higher CLIP score. Extract EVERY visual element from the text and make it visible in the image.
"""
                else:
                    # 第二轮反思时不使用参考图片，所以不需要一致性要求
                    if iteration == 1:
                        consistency_section = """
**Note**: This is the 2nd reflection - generating completely new images based on text content only, no reference images used.
"""
                    else:
                        consistency_section = """
**Consistency Requirements**:
- Maintain the same style, color tone, and atmosphere as the reference image
- If there are characters/pets in previous images, keep their appearance identical
- Change: scene angle, specific objects/details shown, but maintain overall theme
- Ensure variety while keeping semantic consistency with text
"""
                    
                    prompt_text = f"""
Generate image {i+1}/{num_images} for a Xiaohongshu post (continuing from previous image).

**Text Content (MUST match this exactly - extract ALL visual elements):**
{text_content}

{focus_guidance}

{consistency_section}

**CRITICAL Requirements (for optimal CLIP score - this is REFLECTION optimization)**:
1. **Extract and visualize keywords from text**: Identify ALL concrete nouns, colors, actions, scenes, and details in the text and ensure they appear in the image
2. **Visualize text descriptions literally**: If text mentions specific objects, colors, or details, show them exactly
3. **The image MUST visually represent different aspects of the text content** - extract visual elements that haven't been shown in previous images yet
4. **Maintain semantic alignment**: Every element in the image should correspond to something mentioned in the text
5. **ABSOLUTELY NO TEXT OVERLAYS**: Strictly prohibit any text on the image, especially Chinese characters! The image must be pure visual content without any text overlays

**REMEMBER**: This is a REFLECTION optimization - maximize image-text semantic matching for higher CLIP score.
"""
                
                for attempt in range(1, max_retries + 1):
                    try:
                        parts = [{"text": prompt_text}]
                        
                        # 第二轮反思（iteration == 1）时不使用旧图片参考，完全基于文本上下文
                        # 其他轮次可以使用参考图片保持一致性
                        if iteration == 1:
                            # 第二轮反思：不使用任何旧图片参考，只看文本上下文
                            pass  # 不添加任何参考图片
                        elif reference_image and i > 0:
                            # 后续图片：参考前一张新生成的图片保持一致性
                            parts.insert(0, {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": reference_image
                                }
                            })
                        elif reference_image and i == 0:
                            # 第一张图：参考旧图片的风格（仅非第二轮反思时）
                            parts.append({
                                "inlineData": {
                                    "mimeType": "image/png", 
                                    "data": reference_image
                                }
                            })
                            parts.append({"text": "\n(Above is style reference - generate new image matching the text while maintaining similar photographic style)"})
                        
                        resp = requests.post(
                            f"{self.generate_base_url}/models/gemini-2.5-flash-image-preview:generateContent",
                            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.generate_api_key}"},
                            json={
                                "contents": [{"parts": parts}],
                                "generationConfig": {"temperature": 0.6}  # 稍低的temperature以提高一致性
                            },
                            timeout=60
                        )
                        
                        if resp.status_code == 200:
                            result_parts = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
                            for part in result_parts:
                                if "inlineData" in part:
                                    image_data = base64.b64decode(part["inlineData"]["data"])
                                    with open(image_path, "wb") as f:
                                        f.write(image_data)
                                    
                                    new_image_paths.append(image_path)
                                    reference_image = part["inlineData"]["data"]
                                    
                                    print(f"   ✅ 第 {i+1}/{num_images} 张新图片已生成")
                                    success = True
                                    break
                            
                            if success:
                                break  # 成功生成，退出重试循环
                            else:
                                if attempt < max_retries:
                                    print(f"   ⚠️  第 {i+1} 张图片生成失败，重试 {attempt+1}/{max_retries}...")
                        else:
                            print(f"   ⚠️  API错误: {resp.status_code}")
                            if attempt < max_retries:
                                print(f"   🔄 重试 {attempt+1}/{max_retries}...")
                            
                    except Exception as e:
                        print(f"   ❌ 第 {i+1} 张图片生成异常 (尝试 {attempt}/{max_retries}): {str(e)[:100]}")
                        if attempt < max_retries:
                            import time
                            time.sleep(2)  # 等待后重试
                
                # 检查是否成功生成
                if not success:
                    print(f"   ❌ 第 {i+1} 张图片最终生成失败（已重试{max_retries}次）")
                    # 如果是第一张图片失败，后续无法继续
                    if i == 0:
                        print(f"   ⚠️  第一张图片生成失败，无法继续")
                        break
            
            if len(new_image_paths) != num_images:
                print(f"   ⚠️  图片生成不完整（{len(new_image_paths)}/{num_images}），放弃重建")
                return None
            
            print(f"   🎉 成功重新生成 {len(new_image_paths)} 张图片")
            return new_image_paths
            
        except Exception as e:
            print(f"   ❌ 图片重新生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _regenerate_paragraph(self, old_text, issue, suggestion, image_context=""):
        """
        根据issue和suggestion，调用AI重新生成改进后的段落
        
        Args:
            old_text: 原始文本段落
            issue: 问题描述
            suggestion: 改进建议
            image_context: 相关图片的描述（从multimodal分析获取）
            
        Returns:
            改进后的文本，如果失败则返回None
        """
        try:
            # 添加图片上下文（如果有）
            image_context_section = ""
            if image_context:
                image_context_section = f"""
**相关图片描述（参考，确保文字与图片匹配）：**
{image_context}
"""
            
            prompt = f"""You are an AI content optimizer for Xiaohongshu (RedNote) posts. Your goal is to rewrite text to MAXIMIZE semantic consistency with related images while maintaining natural expression.

**Original Text:**
{old_text}

**Identified Issue:**
{issue}

**Improvement Suggestion:**
{suggestion}

{image_context_section}

**Critical Requirements (in order of priority - REFLECTION optimization for higher CLIP score):**
1. **SEMANTIC CONSISTENCY IS PARAMOUNT**: The rewritten text MUST closely match the visual content and semantic meaning of the related images
2. **Use specific, concrete descriptions**: If the image shows specific objects, colors, scenes, or actions, the text MUST explicitly mention them
   - Extract ALL visual elements: objects (e.g., "latte", "wooden table", "white cup"), colors (e.g., "white", "brown"), actions (e.g., "drinking", "holding"), scenes (e.g., "coffee shop", "outdoor")
   - Mention specific details visible in the image: patterns, textures, lighting, composition elements
3. **Keyword alignment**: Extract key visual elements from the image description and naturally incorporate them into the text
   - Add visual keywords that match what's shown in the image
   - Use concrete nouns and descriptive adjectives that correspond to image content
4. **MUST include ultra-specific details** (CRITICAL for quality improvement):
   - For food recommendations: specific dish names, restaurant names, addresses/locations, prices, purchase channels
   - For travel/exploration: specific attraction/shop names, detailed addresses or subway stations, opening hours, ticket prices, personal recommended experiences
   - For products/recommendations: complete product names and models, brand names, purchase channel types, price ranges or discount info
5. **Real experience details**: Describe specific scenarios and feelings, share small details from usage/experience, can mention experiences with friends/family
6. **Practicality first**: Xiaohongshu users love "dry goods" - provide actionable advice, can include Tips, precautions, pitfalls to avoid
7. **Preserve core meaning**: Keep the original intent and information, but express it in a way that better aligns with the image
8. **Natural Xiaohongshu style**: Conversational, engaging, relatable (but secondary to semantic alignment)
9. **Avoid vague descriptions**: Don't say "some shop" or "a dish" - must provide specific names. Don't just say "worth trying" - clearly state where to buy/experience it

**Example of good rewriting (for REFLECTION optimization):**
- Original: "这家咖啡店很棒"
- Image shows: A latte with heart latte art on wooden table, white ceramic cup, warm lighting
- Good rewrite: "这家咖啡店的拿铁真的超棒！桌上的心形拉花看起来就很治愈☕ 白色陶瓷杯配上木桌，暖色调的光线让整个氛围特别温馨"
- Bad rewrite: "姐妹们！这家超赞的咖啡店你们一定要去！" (过度优化风格，忽略图片内容，没有视觉关键词)

**Key principle**: After image regeneration in reflection, the text MUST include ALL visual elements shown in the newly generated image to maximize CLIP score.

**Output Requirements:**
- Output ONLY the rewritten paragraph text, no explanations
- No prefixes like "改写后："
- Text should be directly replaceable in the original position"""

            resp = requests.post(
                f"{self.chat_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.chat_api_key}"},
                json={
                    "model": self.chat_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5,  # Lower temperature for more focused, semantically consistent rewriting
                    "max_tokens": 600  # Increased tokens to support detailed descriptions
                },
                timeout=30
            )
            
            if resp.status_code == 200:
                new_text = resp.json()["choices"][0]["message"]["content"].strip()
                
                # 清理可能的前缀
                prefixes = ["改写后：", "改写后:", "修改后：", "修改后:", "新文：", "新文:"]
                for prefix in prefixes:
                    if new_text.startswith(prefix):
                        new_text = new_text[len(prefix):].strip()
                
                return new_text
            else:
                print(f"         ⚠️  API错误: {resp.status_code}")
                return None
                
        except Exception as e:
            print(f"         ⚠️  重新生成失败: {e}")
            return None
    
    def _regenerate_images_with_suggestions(self, html_path, text_content, user_profile, output_dir, image_modifications, tags=None, rag_examples=None, reflection_iteration=None):
        """
        根据reflection_advisor的建议重新生成图片
        
        Args:
            html_path: 当前HTML文件路径
            text_content: 当前文本内容
            user_profile: 用户画像
            output_dir: 输出目录
            image_modifications: reflection_advisor提供的图片修改建议列表
            tags: 标签列表
            rag_examples: RAG样例列表
            reflection_iteration: reflection迭代次数（0=第一次，用于添加特殊限制）
            
        Returns:
            新图片路径列表，如果失败则返回None
        """
        try:
            print(f"\n🎨 根据Advisor建议重新生成图片...")
            
            # 1. 解析当前HTML获取图片信息
            from bs4 import BeautifulSoup
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            content_div = soup.find('div', class_='post-content')
            
            if not content_div:
                print("   ⚠️  无法找到post-content div")
                return None
            
            # 获取当前图片数量
            current_images = content_div.find_all('div', class_='post-image')
            num_images = len(current_images)
            
            if num_images == 0:
                print("   ⚠️  没有图片需要重新生成")
                return None
            
            print(f"   📸 需要重新生成 {num_images} 张图片")
            
            # 2. 获取当前captions
            current_captions = self._extract_captions_from_html(html_path)
            
            # 3. 构建RAG参考文本
            rag_reference = ""
            if rag_examples:
                rag_texts = [ex.get('content', '')[:200] for ex in rag_examples[:3]]
                rag_reference = f"""
**RAG Top-3 Examples (Reference for style and quality):**
{chr(10).join([f"- Example {i+1}: {text}" for i, text in enumerate(rag_texts)])}
"""
            
            # 4. 生成新图片（整合advisor建议）
            theme, focus_desc = self._get_focus_subject(text_content, user_profile)
            must_people = self._must_include_people(text_content, user_profile)
            must_pets = self._must_include_pets(text_content, user_profile)
            
            # 构建聚焦指引
            if must_people:
                focus_guidance = f"""
📍 主题：{theme}
⚠️ **必须包含人物**（如穿搭展示、健身、自拍等场景）
聚焦对象：人物展示，配合 {focus_desc.lower()}
"""
            elif must_pets:
                focus_guidance = f"""
📍 主题：{theme}
⚠️ **必须包含宠物**
聚焦对象：宠物特写和日常
"""
            else:
                focus_guidance = f"""
📍 主题：{theme}
聚焦对象：{focus_desc}
"""
            
            new_image_paths = []
            
            for i in range(num_images):
                image_path = os.path.join(output_dir, f"personalized_post_{i+1}_reflection.png")
                
                # 获取对应图片的修改建议
                modification = None
                for mod in image_modifications:
                    if mod.get('position') == f'image_{i}':
                        modification = mod
                        break
                
                # 构建包含建议的prompt
                modification_guidance = ""
                if modification:
                    current_issue = modification.get('current_issue', '')
                    suggested_changes = modification.get('suggested_changes', '')
                    modification_guidance = f"""
**Advisor Image Modification Suggestions (CRITICAL - follow these exactly):**
- Current Issue: {current_issue}
- Suggested Changes: {suggested_changes}
"""
                
                # 获取当前caption（用于参考）
                caption_ref = ""
                caption_zh = ""
                caption_en = ""
                if i < len(current_captions) and current_captions[i]:
                    if isinstance(current_captions[i], dict):
                        caption_zh = current_captions[i].get('zh', '')
                        caption_en = current_captions[i].get('en', '')
                        caption_ref = f"Current Caption: {caption_zh} ({caption_en})"
                    else:
                        caption_ref = f"Current Caption: {current_captions[i]}"
                        caption_zh = str(current_captions[i])
                
                tags_section = ""
                if tags:
                    tags_str = ", ".join(tags)
                    tags_section = f"""
**Tags (Primary Reference):**
{tags_str}
"""
                
                # 第一次reflection的特殊要求：突出caption作为主体
                caption_emphasis = ""
                if reflection_iteration == 0 and caption_zh:
                    caption_emphasis = f"""
⚠️ **CRITICAL FOR FIRST REFLECTION - CAPTION AS MAIN SUBJECT:**
- **The caption "{caption_zh}" MUST be the PRIMARY and DOMINANT subject in the image**
- The caption object/item should occupy the CENTER and FOREGROUND of the composition
- Other elements (background, context, etc.) should be SECONDARY and support the caption subject
- The image should clearly and prominently show what the caption describes
- Composition priority: Caption subject (70%+) > Context elements (30%-)
- This ensures maximum CLIP score matching between image and caption
"""
                
                prompt_text = f"""
Generate image {i+1}/{num_images} for a Xiaohongshu post (Reflection - regenerate based on context, caption, and advisor suggestions).

{tags_section}

**Text Context (PRIMARY Reference):**
{text_content[:800]}

{caption_ref}

{caption_emphasis}

{modification_guidance}

**User Profile:**
{user_profile[:300]}

{rag_reference}

{focus_guidance}

**CRITICAL Requirements:**
1. **PRIMARY References**: Context text + Caption (most important), RAG examples + user_profile (secondary)
2. **Follow Advisor Suggestions**: Implement the suggested changes exactly as specified
3. **Caption Alignment**: The image must match the current caption description
4. **Natural Xiaohongshu photo style**: Realistic, life-like, social media aesthetic
5. **Clear composition**: Main subject should be prominent and clearly visible
6. **ABSOLUTELY NO TEXT OVERLAYS**: Strictly prohibit any text on the image, especially Chinese characters! The image must be pure visual content without any text overlays

**REMEMBER**: This is reflection optimization - regenerate images based on context+caption (PRIMARY) and advisor suggestions to improve image-caption matching.
"""
                
                # 调用图片生成API（复用generate_images中的逻辑）
                success = False
                max_retries = 3
                reference_image = None  # 用于保持一致性
                
                for attempt in range(1, max_retries + 1):
                    try:
                        if attempt > 1:
                            print(f"   🔄 图片 {i+1} 重试 {attempt}/{max_retries}...")
                        else:
                            print(f"   🎨 生成图片 {i+1}/{num_images}...")
                        
                        # 调用Gemini API生成图片（使用REST API，与generate_images一致）
                        # 构建请求内容
                        parts = [{"text": prompt_text}]
                        
                        # 如果有参考图片且不是第一张，添加参考
                        if reference_image and i > 0:
                            parts.insert(0, {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": reference_image
                                }
                            })
                        
                        resp = requests.post(
                            f"{self.generate_base_url}/models/gemini-2.5-flash-image-preview:generateContent",
                            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.generate_api_key}"},
                            json={
                                "contents": [{"parts": parts}],
                                "generationConfig": {"temperature": 0.9}
                            },
                            timeout=60
                        )
                        
                        if resp.status_code == 200:
                            result_parts = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
                            for part in result_parts:
                                if "inlineData" in part:
                                    # 保存图片
                                    image_data = base64.b64decode(part["inlineData"]["data"])
                                    with open(image_path, "wb") as f:
                                        f.write(image_data)
                                    
                                    if os.path.exists(image_path):
                                        new_image_paths.append(image_path)
                                        success = True
                                        print(f"      ✅ 图片 {i+1} 生成成功")
                                        
                                        # 保存为下一张的参考
                                        reference_image = part["inlineData"]["data"]
                                        break
                        else:
                            print(f"      ⚠️  API返回错误: {resp.status_code}")
                            if resp.text:
                                error_msg = resp.text[:200]
                                print(f"         错误信息: {error_msg}")
                        
                        if success:
                            break
                            
                    except Exception as e:
                        print(f"      ⚠️  尝试 {attempt} 失败: {e}")
                        if attempt == max_retries:
                            print(f"      ❌ 图片 {i+1} 生成失败（已重试{max_retries}次）")
                
                if not success:
                    print(f"   ⚠️  图片 {i+1} 生成失败，跳过")
            
            if new_image_paths:
                print(f"   ✅ 成功生成 {len(new_image_paths)} 张图片")
                return new_image_paths
            else:
                print(f"   ❌ 所有图片生成失败")
                return None
                
        except Exception as e:
            print(f"   ❌ 重新生成图片过程出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _regenerate_images_with_new_captions(self, html_path, text_content, user_profile, output_dir, new_captions, tags=None, rag_examples=None):
        """
        根据新生成的caption和部分上下文重新生成图片（用于第三次reflection）
        
        Args:
            html_path: 当前HTML文件路径
            text_content: 当前文本内容
            user_profile: 用户画像
            output_dir: 输出目录
            new_captions: 新生成的caption列表，每个是dict: {{"zh": "中文", "en": "English"}}
            tags: 标签列表
            rag_examples: RAG样例列表
            
        Returns:
            新图片路径列表，如果失败则返回None
        """
        try:
            print(f"\n🎨 根据新caption和部分上下文重新生成图片...")
            
            # 1. 解析当前HTML获取图片信息
            from bs4 import BeautifulSoup
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            content_div = soup.find('div', class_='post-content')
            
            if not content_div:
                print("   ⚠️  无法找到post-content div")
                return None
            
            # 获取当前图片数量
            current_images = content_div.find_all('div', class_='post-image')
            num_images = len(current_images)
            
            if num_images == 0:
                print("   ⚠️  没有图片需要重新生成")
                return None
            
            print(f"   📸 需要重新生成 {num_images} 张图片")
            
            # 2. 构建RAG参考文本
            rag_reference = ""
            if rag_examples:
                rag_texts = [ex.get('content', '')[:200] for ex in rag_examples[:3]]
                rag_reference = f"""
**RAG Top-3 Examples (Reference for style and quality):**
{chr(10).join([f"- Example {i+1}: {text}" for i, text in enumerate(rag_texts)])}
"""
            
            # 3. 生成新图片（根据新caption和部分上下文）
            theme, focus_desc = self._get_focus_subject(text_content, user_profile)
            must_people = self._must_include_people(text_content, user_profile)
            must_pets = self._must_include_pets(text_content, user_profile)
            
            # 构建聚焦指引
            if must_people:
                focus_guidance = f"""
📍 主题：{theme}
⚠️ **必须包含人物**（如穿搭展示、健身、自拍等场景）
聚焦对象：人物展示，配合 {focus_desc.lower()}
"""
            elif must_pets:
                focus_guidance = f"""
📍 主题：{theme}
⚠️ **必须包含宠物**
聚焦对象：宠物特写和日常
"""
            else:
                focus_guidance = f"""
📍 主题：{theme}
聚焦对象：{focus_desc}
"""
            
            new_image_paths = []
            reference_image = None  # 用于保持一致性
            
            for i in range(num_images):
                image_path = os.path.join(output_dir, f"personalized_post_{i+1}_reflection3.png")
                
                # 获取对应的新caption
                caption_dict = None
                if i < len(new_captions):
                    caption_dict = new_captions[i]
                
                caption_zh = caption_dict.get('zh', '') if caption_dict else ''
                caption_en = caption_dict.get('en', '') if caption_dict else ''
                
                tags_section = ""
                if tags:
                    tags_str = ", ".join(tags)
                    tags_section = f"""
**Tags:**
{tags_str}
"""
                
                # 构建prompt：强调caption作为主体，部分上下文作为辅助
                prompt_text = f"""
Generate image {i+1}/{num_images} for a Xiaohongshu post (3rd Reflection - regenerate based on NEW caption and partial context).

**NEW Caption (PRIMARY - MUST be the main subject):**
- Chinese: {caption_zh}
- English: {caption_en}

⚠️ **CRITICAL: The caption "{caption_en}" MUST be the PRIMARY and DOMINANT subject in the image**
- The caption object/item should occupy the CENTER and FOREGROUND of the composition
- Other elements should be SECONDARY and support the caption subject
- Composition priority: Caption subject (70%+) > Context elements (30%-)

**Partial Text Context (SECONDARY - for style and atmosphere only):**
{text_content[:400]}

{tags_section}

**User Profile:**
{user_profile[:300]}

{rag_reference}

{focus_guidance}

**CRITICAL Requirements:**
1. **Caption as Main Subject**: The caption "{caption_en}" MUST be the dominant visual element
2. **Context for Style**: Use partial context only for style, atmosphere, and background elements
3. **Natural Xiaohongshu photo style**: Realistic, life-like, social media aesthetic
4. **Clear composition**: Caption subject should be prominent and clearly visible
5. **ABSOLUTELY NO TEXT OVERLAYS**: Strictly prohibit any text on the image, especially Chinese characters! The image must be pure visual content without any text overlays

**REMEMBER**: This is 3rd reflection - regenerate images with caption as PRIMARY subject, partial context as SECONDARY reference.
"""
                
                # 调用图片生成API
                success = False
                max_retries = 3
                
                for attempt in range(1, max_retries + 1):
                    try:
                        if attempt > 1:
                            print(f"   🔄 图片 {i+1} 重试 {attempt}/{max_retries}...")
                        else:
                            print(f"   🎨 生成图片 {i+1}/{num_images}...")
                        
                        # 调用Gemini API生成图片（使用REST API）
                        parts = [{"text": prompt_text}]
                        
                        # 如果有参考图片且不是第一张，添加参考
                        if reference_image and i > 0:
                            parts.insert(0, {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": reference_image
                                }
                            })
                        
                        resp = requests.post(
                            f"{self.generate_base_url}/models/gemini-2.5-flash-image-preview:generateContent",
                            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.generate_api_key}"},
                            json={
                                "contents": [{"parts": parts}],
                                "generationConfig": {"temperature": 0.9}
                            },
                            timeout=60
                        )
                        
                        if resp.status_code == 200:
                            result_parts = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
                            for part in result_parts:
                                if "inlineData" in part:
                                    # 保存图片
                                    image_data = base64.b64decode(part["inlineData"]["data"])
                                    with open(image_path, "wb") as f:
                                        f.write(image_data)
                                    
                                    if os.path.exists(image_path):
                                        new_image_paths.append(image_path)
                                        success = True
                                        print(f"      ✅ 图片 {i+1} 生成成功")
                                        
                                        # 保存为下一张的参考
                                        reference_image = part["inlineData"]["data"]
                                        break
                        else:
                            print(f"      ⚠️  API返回错误: {resp.status_code}")
                            if resp.text:
                                error_msg = resp.text[:200]
                                print(f"         错误信息: {error_msg}")
                        
                        if success:
                            break
                            
                    except Exception as e:
                        print(f"      ⚠️  尝试 {attempt} 失败: {e}")
                        if attempt == max_retries:
                            print(f"      ❌ 图片 {i+1} 生成失败（已重试{max_retries}次）")
                
                if not success:
                    print(f"   ⚠️  图片 {i+1} 生成失败，跳过")
            
            if new_image_paths:
                print(f"   ✅ 成功生成 {len(new_image_paths)} 张图片")
                return new_image_paths
            else:
                print(f"   ❌ 所有图片生成失败")
                return None
                
        except Exception as e:
            print(f"   ❌ 重新生成图片过程出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _replace_images_in_html(self, html_path, new_image_paths, iteration):
        """
        替换HTML中的图片路径（保留原有caption）
        
        Args:
            html_path: 当前HTML文件路径
            new_image_paths: 新图片路径列表
            iteration: 当前迭代次数
            
        Returns:
            新HTML文件路径，如果失败则返回None
        """
        try:
            print(f"\n   🔄 更新HTML中的图片引用（保留caption）...")
            
            # 读取HTML
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            content_div = soup.find('div', class_='post-content')
            
            if not content_div:
                print("   ⚠️  无法找到post-content div")
                return None
            
            # 查找所有图片div
            image_divs = content_div.find_all('div', class_='post-image')
            
            if len(image_divs) != len(new_image_paths):
                print(f"   ⚠️  图片数量不匹配：HTML中{len(image_divs)}张，新图片{len(new_image_paths)}张")
                return None
            
            # 替换每张图片的src（保留caption）
            for idx, (img_div, new_img_path) in enumerate(zip(image_divs, new_image_paths)):
                img_tag = img_div.find('img')
                if img_tag:
                    # 使用相对路径
                    new_img_filename = os.path.basename(new_img_path)
                    img_tag['src'] = new_img_filename
                    # caption保持不变
                    caption_div = img_div.find('div', class_='image-caption')
                    caption_text = caption_div.get_text(strip=True) if caption_div else f"图 {idx+1}"
                    print(f"      ✅ 图片 {idx+1}: {new_img_filename} (caption: {caption_text[:20]}...)")
            
            # 保存为新版本
            output_dir = Path(html_path).parent
            new_html_path = output_dir / f"image_text_v{iteration+1}.html"
            
            with open(new_html_path, 'w', encoding='utf-8') as f:
                f.write(str(soup))
            
            print(f"   ✅ 新版本HTML已保存: {new_html_path.name}")
            return str(new_html_path)
            
        except Exception as e:
            print(f"   ❌ HTML更新失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _apply_reflection_suggestions(self, html_path, suggestions, iteration):
        """
        应用Reflection建议，生成改进版本的HTML
        
        Args:
            html_path: 原HTML文件路径
            suggestions: Reflection建议（包含text_changes和image_captions）
            iteration: 当前迭代次数（1-3）
            
        Returns:
            新HTML文件路径，如果失败则返回None
        """
        try:
            print(f"\n🔧 应用Reflection建议（第{iteration}次优化）...")
            
            # 读取原HTML
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            content_div = soup.find('div', class_='post-content')
            
            if not content_div:
                print("⚠️  无法找到post-content div")
                return None
            
            # 1. 应用图片caption修改
            if suggestions.get('image_captions'):
                print(f"   📸 更新 {len(suggestions['image_captions'])} 个图片caption...")
                for caption_suggestion in suggestions['image_captions']:
                    position = caption_suggestion.get('position', '')  # 如 "image_0"
                    new_caption = caption_suggestion.get('caption', '')  # 如 "图1: 简短描述"
                    
                    if position.startswith('image_'):
                        try:
                            image_index = int(position.split('_')[1])
                            # 查找对应的图片caption
                            image_divs = content_div.find_all('div', class_='post-image')
                            if image_index < len(image_divs):
                                caption_div = image_divs[image_index].find('div', class_='image-caption')
                                if caption_div:
                                    # 处理新的dict格式或旧的字符串格式
                                    if isinstance(new_caption, dict):
                                        caption_zh = new_caption.get("zh", "")
                                        caption_en = new_caption.get("en", "")
                                        formatted_caption = f"图{image_index + 1}: {caption_zh}"
                                        # 设置data-caption-en属性（用于CLIP计算）
                                        caption_div['data-caption-en'] = caption_en
                                        caption_div.string = formatted_caption
                                        print(f"      ✅ {position}: {formatted_caption} ({caption_en})")
                                    else:
                                        # 向后兼容：如果是字符串，假设是英文
                                        if not new_caption.startswith("图"):
                                            formatted_caption = f"图{image_index + 1}: {new_caption}"
                                        else:
                                            formatted_caption = new_caption
                                        caption_div['data-caption-en'] = new_caption
                                        caption_div.string = formatted_caption
                                        print(f"      ✅ {position}: {formatted_caption}")
                        except Exception as e:
                            print(f"      ⚠️  更新caption失败 ({position}): {e}")
            
            # 2. 应用文本修改（调用AI重新生成改进后的段落）
            if suggestions.get('text_changes'):
                # 过滤掉link相关的建议（用户要求先不管link）
                text_changes = [c for c in suggestions['text_changes'] if not c.get('position', '').startswith('link_')]
                
                # 准备图片caption信息（用作上下文）
                image_analyses = suggestions.get('image_analyses', [])
                image_captions_map = {}
                for img_analysis in image_analyses:
                    img_index = img_analysis.get('image_index', -1)
                    if img_index >= 0:
                        image_captions_map[img_index] = img_analysis.get('caption', '')
                
                if text_changes:
                    print(f"   ✍️  应用 {len(text_changes)} 条文本修改建议...")
                    
                    # 提取所有<p>标签和<div class="post-image">（用于定位相邻图片）
                    all_elements = []
                    for elem in content_div.find_all(['p', 'div'], recursive=False):
                        if elem.name == 'p':
                            all_elements.append(('text', elem))
                        elif elem.name == 'div' and 'post-image' in elem.get('class', []):
                            all_elements.append(('image', elem))
                    
                    # 构建text_index到element的映射
                    text_elements = []
                    text_index_to_position = {}
                    text_counter = 0
                    for i, (elem_type, elem) in enumerate(all_elements):
                        if elem_type == 'text':
                            text_elements.append((elem, i))
                            text_index_to_position[text_counter] = i
                            text_counter += 1
                    
                    for change in text_changes:
                        position = change.get('position', '')  # 如 "text_0"
                        issue = change.get('issue', '')
                        suggestion_text = change.get('suggestion', '')
                        
                        # 解析position（如 "text_0" -> index 0）
                        if position.startswith('text_'):
                            try:
                                text_index = int(position.split('_')[1])
                                
                                if text_index < len(text_elements):
                                    old_paragraph, elem_position = text_elements[text_index]
                                    old_text = old_paragraph.get_text(strip=True)
                                    
                                    # 查找相邻的图片caption（作为上下文）
                                    image_context = ""
                                    # 查找前后的图片
                                    for offset in [-1, 1, -2, 2]:
                                        check_pos = elem_position + offset
                                        if 0 <= check_pos < len(all_elements):
                                            elem_type, elem = all_elements[check_pos]
                                            if elem_type == 'image':
                                                # 尝试找到这个图片的caption
                                                img_caption_div = elem.find('div', class_='image-caption')
                                                if img_caption_div:
                                                    caption_text = img_caption_div.get_text(strip=True)
                                                    if caption_text and caption_text != f"图 {offset}":
                                                        image_context += f"- {caption_text}\n"
                                    
                                    # 调用AI重新生成改进后的段落（传入图片上下文）
                                    print(f"      🔄 {position}: 正在重新生成...")
                                    if image_context:
                                        print(f"         📸 相关图片: {image_context[:60]}...")
                                    
                                    new_text = self._regenerate_paragraph(
                                        old_text=old_text,
                                        issue=issue,
                                        suggestion=suggestion_text,
                                        image_context=image_context if image_context else ""
                                    )
                                    
                                    if new_text and new_text != old_text:
                                        # 替换段落内容（保留HTML标签结构）
                                        old_paragraph.clear()
                                        # 处理markdown（加粗等）
                                        rendered_text = self._render_markdown(new_text)
                                        from bs4 import BeautifulSoup as BS
                                        rendered_soup = BS(rendered_text, 'html.parser')
                                        for child in rendered_soup.children:
                                            old_paragraph.append(child)
                                        
                                        print(f"      ✅ {position}: 已更新")
                                        print(f"         原文: {old_text[:50]}...")
                                        print(f"         新文: {new_text[:50]}...")
                                    else:
                                        print(f"      ⚠️  {position}: AI未生成新内容或内容相同，跳过")
                                        
                            except Exception as e:
                                print(f"      ⚠️  {position}: 修改失败 - {e}")
                                import traceback
                                traceback.print_exc()
            
            # 3. 保存为新版本（或覆盖现有版本）
            output_dir = Path(html_path).parent
            target_version = f"_v{iteration+1}"  # v2, v3, v4
            new_html_path = output_dir / f"image_text{target_version}.html"
            
            # 检查当前html_path是否已经是目标版本（例如图片重建后）
            current_filename = Path(html_path).name
            if current_filename == f"image_text{target_version}.html":
                # 已经是目标版本，直接覆盖
                print(f"   🔄 覆盖现有版本: {current_filename}（图片+文本/caption联合优化）")
            else:
                # 创建新版本
                print(f"   ✅ 创建新版本: {new_html_path.name}")
            
            with open(new_html_path, 'w', encoding='utf-8') as f:
                f.write(str(soup))
            
            return str(new_html_path)
            
        except Exception as e:
            print(f"   ❌ 应用建议失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def __call__(self, user_profile, output_dir, user_profile_path=None):
        """Main execution flow - 支持 RAG 模式
        
        Args:
            user_profile: User profile text or dict
            output_dir: Output directory
            user_profile_path: Path to user profile file (for RAG)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract product ID from output_dir (format: generated_it/{index}_{user_id}/)
        product_id = os.path.basename(os.path.normpath(output_dir))
        # Extract index if available (format: {index}_{user_id})
        product_index = None
        if '_' in product_id:
            try:
                product_index = int(product_id.split('_')[0])
            except ValueError:
                pass
        
        # Parse profile data if needed
        profile_data = None
        if user_profile_path and os.path.exists(user_profile_path):
            try:
                with open(user_profile_path, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                print(f"📋 Loaded profile from: {user_profile_path}")
            except Exception as e:
                print(f"⚠️ Failed to load profile data: {e}")
        
        # Get profile text
        if isinstance(user_profile, dict):
            profile_text = user_profile.get("profile_text", json.dumps(user_profile, ensure_ascii=False))
        else:
            profile_text = str(user_profile)
            if profile_data is None:
                profile_data = {"profile_text": profile_text}
        
        print("1. 生成文案、标签与平台链接...")
        text, tags, links = self.generate_text(profile_text, 
                                               user_profile_path=user_profile_path,
                                               profile_data=profile_data)
        print(f"   - 提取到 {len(tags)} 个标签: {tags}")
        print(f"   - 提取到 {len(links)} 个推荐链接: {[l['platform'] for l in links]}")
        
        print("2. 提取caption（基于文案，不分析图片）...")
        # 🎯 先基于文案提取caption，这样初产品分数不会太高，reflection效果才能体现
        # 计算需要多少张图片
        word_count = len(text.strip())
        num_images = 1 if word_count <= 300 else (2 if word_count <= 800 else 3)
        
        # 一次性提取所有图片的caption，确保它们不同
        image_captions = self._extract_all_captions_from_text(text, num_images)
        for i, caption in enumerate(image_captions):
            if isinstance(caption, dict):
                print(f"   📝 图片{i+1} caption: {caption.get('zh', '')} ({caption.get('en', '')})")
            else:
                print(f"   📝 图片{i+1} caption: {caption}")
        
        print("3. 生成配图...")
        # 第二轮反思时会使用完整文案上下文重新生成图片
        images = self.generate_images(profile_text, text, output_dir, tags=tags, image_captions=image_captions)
        
        # 检查图片生成情况
        initial_image_generation_failed = len(images) == 0
        if initial_image_generation_failed:
            print(f"⚠️  初始图片生成失败（0张图片）")
            print(f"💡 策略：先生成纯文本HTML，在Reflection时尝试重新生成图片")
            image_captions = []  # 确保captions为空
        else:
            print(f"✅ 成功生成 {len(images)} 张图片")
            print(f"   📝 使用预提取的Captions: {image_captions}")
        
        print("4. 生成HTML...")
        html_path = os.path.join(output_dir, "image_text_v0.html")  # 初始版本为v0
        self.generate_html_post(text, images, links, tags, html_path, image_captions=image_captions)
        print(f"✅ 完成: {html_path}")
        
        # 4. Reflection机制（自动迭代优化）
        reflection_history = []  # 记录每次reflection的结果
        current_html_path = html_path
        final_version = "v0"  # 初始版本为v0
        
        if self.reflection_enabled:
            print(f"\n{'='*80}")
            print(f"🔄 启动Reflection机制（最多{self.max_reflection_iterations}次迭代）")
            print(f"   阈值: GroupScore ≥ {self.reflection_threshold}")
            print(f"{'='*80}")
            
            for iteration in range(self.max_reflection_iterations):
                print(f"\n{'─'*80}")
                # Display product ID/index in evaluation header
                if product_index is not None:
                    print(f"📊 [{product_index}] 第{iteration+1}次评估 (当前版本: v{iteration})")
                else:
                    print(f"📊 [{product_id}] 第{iteration+1}次评估 (当前版本: v{iteration})")
                print(f"{'─'*80}")
                
                try:
                    # 4.1 计算GroupScore
                    print(f"\n1️⃣  计算GroupScore...")
                    eval_result = evaluate_file(
                        html_path=current_html_path,
                        evaluator=self.clip_evaluator,
                        use_combined=True,
                        verbose=False  # 不打印详细信息，保持输出简洁
                    )
                    
                    if not eval_result:
                        print(f"   ⚠️  GroupScore计算失败，跳过reflection")
                        break
                    
                    # 🚨 特殊处理：如果没有图片（初始生成失败），立即尝试生成
                    if eval_result.num_pairs == 0:
                        print(f"   ⚠️  检测到没有图片（num_pairs=0）")
                        print(f"   💡 尝试生成图片以修复问题...")
                        
                        try:
                            # 尝试生成图片
                            new_image_paths = self._regenerate_images_for_reflection(
                                html_path=current_html_path,
                                text_content=text,
                                user_profile=profile_text,
                                output_dir=output_dir,
                                iteration=iteration
                            )
                            
                            if new_image_paths and len(new_image_paths) > 0:
                                # 重新生成HTML（添加图片）
                                print(f"   🔄 重新生成HTML（添加图片）...")
                                new_html_path = os.path.join(output_dir, f"image_text_v{iteration+1}.html")
                                # 重新生成图片时保留原有caption（从当前HTML中提取）
                                current_captions = self._extract_captions_from_html(html_path)
                                self.generate_html_post(text, new_image_paths, links, tags, new_html_path, image_captions=current_captions)
                                
                                # 验证新版本
                                new_eval_result = evaluate_file(
                                    html_path=new_html_path,
                                    evaluator=self.clip_evaluator,
                                    use_combined=True,
                                    verbose=False
                                )
                                
                                if new_eval_result and new_eval_result.num_pairs > 0:
                                    groupscore = new_eval_result.group_score_mean
                                    print(f"   ✅ 成功添加图片！GroupScore (Mean): {groupscore:.4f}")
                                    current_html_path = new_html_path
                                    final_version = f"v{iteration+1}"
                                    
                                    # 如果达到阈值，记录并结束
                                    if groupscore >= self.reflection_threshold:
                                        print(f"   🎉 GroupScore已达到阈值！")
                                        # 达标时才记录，因为不会再有下一个iteration
                                        reflection_history.append({
                                            "iteration": iteration + 1,
                                            "version": f"v{iteration+1}",
                                            "groupscore": groupscore,
                                            "html_path": new_html_path,
                                            "strategy": "image_generation_rescue"
                                        })
                                        break
                                    else:
                                        # 未达标，继续下一次迭代
                                        # 下一个iteration开始时会自动评估和记录
                                        continue
                                else:
                                    print(f"   ⚠️  添加图片后评估失败，停止reflection")
                                    break
                            else:
                                print(f"   ❌ 图片生成仍然失败，无法继续reflection")
                                break
                        except Exception as e:
                            print(f"   ❌ 图片生成补救失败: {e}")
                            break
                    
                    # 使用mean作为主要评估指标
                    groupscore = eval_result.group_score_mean
                    print(f"   📈 GroupScore (Mean): {groupscore:.4f}")
                    print(f"      (Harmonic: {eval_result.group_score_harmonic:.4f}, Min: {eval_result.group_score_min:.4f})")
                    
                    # 记录本次评估（但要避免重复记录）
                    # 检查history中是否已有相同版本的记录（可能在上一iteration的图片重建中记录过）
                    current_version = f"v{iteration}"
                    already_recorded = any(
                        record['version'] == current_version 
                        for record in reflection_history
                    )
                    
                    if not already_recorded:
                        reflection_history.append({
                            "iteration": iteration + 1,
                            "version": current_version,
                            "groupscore": groupscore,
                            "html_path": current_html_path
                        })
                    else:
                        print(f"   ℹ️  版本{current_version}已在history中，跳过重复记录")
                    
                    # 4.2 判断是否达到阈值
                    if groupscore >= self.reflection_threshold:
                        print(f"   ✅ GroupScore达标！无需进一步优化")
                        final_version = f"v{iteration}"
                        break
                    
                    print(f"   ⚠️  GroupScore ({groupscore:.4f}) < 阈值 ({self.reflection_threshold})")
                    print(f"   🔄 启动第{iteration+1}次Reflection...")
                    
                    # 4.3 解析HTML
                    print(f"\n2️⃣  解析HTML...")
                    parse_result = self.html_parser.parse_html_to_sequence(current_html_path)
                    html_sequence = parse_result["sequence_text"]
                    image_paths = parse_result["image_paths"]
                    print(f"   ✅ 解析完成: {parse_result['stats']['texts']} 文本, {parse_result['stats']['images']} 图片")
                    
                    # 4.4 获取RAG样例（复用已加载的examples）
                    print(f"\n3️⃣  准备RAG样例...")
                    rag_examples = []
                    
                    if hasattr(self, '_rag_cache') and self._rag_cache:
                        # 使用缓存的RAG样例
                        cache_key = list(self._rag_cache.keys())[0]
                        cached_examples = self._rag_cache[cache_key]
                        
                        # 只提取文本样例（过滤掉图片）
                        text_examples = [ex for ex in cached_examples if ex.get("type") == "text"]
                        
                        for ex in text_examples[:3]:  # Top-3 文本样例
                            rag_examples.append({
                                "content": ex.get("content", ""),
                                "similarity": ex.get("similarity", 0)
                            })
                        print(f"   ✅ 使用 {len(rag_examples)} 个缓存的RAG文本样例")
                    else:
                        # RAG缓存为空，尝试多种方式获取
                        print(f"   ⚠️  RAG缓存为空，尝试其他方式获取样例...")
                        
                        # 方式1: 如果有profile路径，重新加载
                        if user_profile_path and profile_data:
                            try:
                                user_id = self.extract_user_id_from_path(user_profile_path)
                                if user_id:
                                    top1_preference = self.extract_top1_preference(profile_data)
                                    examples = self.load_examples_with_rag(user_id, top1_preference, top_k=3)
                                    
                                    # 提取文本样例
                                    text_examples = [ex for ex in examples if ex.get("type") == "text"]
                                    for ex in text_examples[:3]:
                                        rag_examples.append({
                                            "content": ex.get("content", ""),
                                            "similarity": ex.get("similarity", 0)
                                        })
                                    print(f"   ✅ 重新加载成功，获取 {len(rag_examples)} 个RAG文本样例")
                            except Exception as e:
                                print(f"   ⚠️  重新加载失败: {e}")
                        
                        # 方式2: 如果方式1失败，尝试从fallback目录加载
                        if not rag_examples:
                            try:
                                fallback_examples = self._load_examples_fallback()
                                text_examples = [ex for ex in fallback_examples if ex.get("type") == "text"]
                                for ex in text_examples[:3]:
                                    rag_examples.append({
                                        "content": ex.get("content", ""),
                                        "similarity": 0.5  # 默认相似度
                                    })
                                if rag_examples:
                                    print(f"   ✅ 从fallback目录加载 {len(rag_examples)} 个样例")
                            except Exception as e:
                                print(f"   ⚠️  Fallback加载失败: {e}")
                        
                        # 如果还是没有样例，就使用空列表（不影响功能）
                        if not rag_examples:
                            print(f"   ℹ️  无可用RAG样例，Reflection将仅基于当前内容和图片分析")
                    
                    # 4.5 调用Reflection Advisor
                    print(f"\n4️⃣  AI Reflection分析...")
                    
                    # 初始化变量
                    should_apply = False
                    suggestion_detail = {}
                    
                    # 🎯 第一次Reflection：改图片（根据上下文+caption+RAG+user_profile）
                    if iteration == 0:
                        print(f"   📝 策略：第1次Reflection - 根据上下文+caption+RAG+user_profile重新生成图片")
                        
                        # 获取当前captions
                        current_captions = self._extract_captions_from_html(current_html_path)
                        
                        # 调用reflection_advisor获取图片修改建议
                        advisor_result = self.reflection_advisor.evaluate_and_suggest(
                            groupscore=groupscore,
                            html_sequence=html_sequence,
                            rag_examples=rag_examples,
                            image_paths=image_paths,
                            threshold=self.reflection_threshold,
                            user_profile=profile_text,
                            reflection_iteration=0
                        )
                        
                        if advisor_result.get('should_modify') and advisor_result.get('suggestions'):
                            image_modifications = advisor_result.get('suggestions', {}).get('image_modifications', [])
                            
                            if image_modifications:
                                print(f"   ✅ 获得 {len(image_modifications)} 条图片修改建议")
                                # 根据建议重新生成图片（第一次reflection需要突出caption主体）
                                new_image_paths = self._regenerate_images_with_suggestions(
                                    html_path=current_html_path,
                                    text_content=text,
                                    user_profile=profile_text,
                                    output_dir=output_dir,
                                    image_modifications=image_modifications,
                                    tags=tags,
                                    rag_examples=rag_examples,
                                    reflection_iteration=0  # 标识是第一次reflection
                                )
                                
                                if new_image_paths:
                                    # 替换HTML中的图片（保留caption）
                                    new_html_path = self._replace_images_in_html(
                                        html_path=current_html_path,
                                        new_image_paths=new_image_paths,
                                        iteration=iteration
                                    )
                                    
                                    if new_html_path:
                                        # 验证新图片的效果
                                        print(f"\n   🔍 验证新图片的效果...")
                                        try:
                                            new_eval_result = evaluate_file(
                                                html_path=new_html_path,
                                                evaluator=self.clip_evaluator,
                                                use_combined=True,
                                                verbose=False
                                            )
                                            new_score = new_eval_result.group_score_mean
                                            score_improvement = new_score - groupscore
                                            
                                            print(f"      📊 重新生成前 (Mean): {groupscore:.4f}")
                                            print(f"      📊 重新生成后 (Mean): {new_score:.4f}")
                                            print(f"      📊 提升幅度: {score_improvement:+.4f}")
                                            
                                            if new_score > groupscore or not self.reflection_strict_mode:
                                                print(f"      ✅ 图片重新生成成功！")
                                                current_html_path = new_html_path
                                                groupscore = new_score
                                                final_version = f"v{iteration+1}"
                                                
                                                if new_score >= self.reflection_threshold:
                                                    print(f"      🎉 Score已达到阈值！")
                                                    reflection_history.append({
                                                        "iteration": iteration + 1,
                                                        "version": f"v{iteration+1}",
                                                        "groupscore": new_score,
                                                        "html_path": new_html_path,
                                                        "strategy": "image_regeneration_1st_reflection"
                                                    })
                                                    break
                                                else:
                                                    print(f"      ✅ 第1次反思完成，进入下一iteration")
                                                    continue
                                            else:
                                                print(f"      ⚠️  图片重新生成后Score未提升，保留原图片")
                                                print(f"      ✅ 第1次反思完成，进入下一iteration")
                                                continue
                                        except Exception as e:
                                            print(f"      ⚠️  验证失败: {e}")
                                            continue
                                    else:
                                        print(f"   ⚠️  HTML更新失败，继续使用原图片")
                                        continue
                                else:
                                    print(f"   ⚠️  图片重新生成失败，继续使用原图片")
                                    continue
                            else:
                                print(f"   ⚠️  未获得图片修改建议，跳过")
                                continue
                        else:
                            print(f"   ℹ️  Advisor建议无需修改，跳过")
                            continue
                    
                    # 🎯 第二次Reflection：根据图片调整caption，确保caption描述的是图片主体
                    elif iteration == 1:
                        print(f"   📝 策略：第2次Reflection - 根据图片主体调整caption")
                        print(f"   🎯 目标：确保caption准确描述图片中的主要视觉元素")
                        
                        # 直接使用vision model分析图片，生成基于图片主体的caption
                        new_captions = []
                        for idx, img_path in enumerate(image_paths):
                            if img_path and os.path.exists(img_path):
                                print(f"   🔍 分析图片 {idx+1}/{len(image_paths)}...")
                                # 使用_extract_keywords_for_caption方法，基于图片内容生成caption
                                new_caption = self._extract_keywords_for_caption(img_path, text_content=text)
                                new_captions.append({
                                    "position": f"image_{idx}",
                                    "caption": new_caption
                                })
                                if isinstance(new_caption, dict):
                                    print(f"      📸 图片{idx}: {new_caption.get('zh', '')} ({new_caption.get('en', '')})")
                                else:
                                    print(f"      📸 图片{idx}: {new_caption}")
                            else:
                                print(f"      ⚠️  图片{idx}不存在，跳过")
                        
                        if new_captions:
                            print(f"   ✅ 根据图片主体生成了 {len(new_captions)} 个新caption")
                            should_apply = True
                            suggestion_detail = {
                                "text_changes": [],
                                "image_captions": new_captions
                            }
                        else:
                            print(f"   ⚠️  无法生成caption，跳过")
                            continue
                    
                    # 🎯 第三次及后续Reflection：先重新生成简短caption，再根据caption及部分上下文生成图片
                    # iteration == 2: 第3次反思（特定策略）
                    # iteration >= 3: 第4次及以后，都使用第3次的策略
                    elif iteration >= 2:
                        iteration_num = iteration + 1
                        print(f"   📝 策略：第{iteration_num}次Reflection - 先生成简短caption，再根据caption+上下文生成图片")
                        if iteration > 2:
                            print(f"   ℹ️  使用第3次反思的策略（重复执行直到达到阈值）")
                        
                        # Step 1: 先根据当前图片生成新的简短caption
                        print(f"\n   📝 Step 1: 根据当前图片生成新的简短caption...")
                        new_captions = []
                        for idx, img_path in enumerate(image_paths):
                            if img_path and os.path.exists(img_path):
                                print(f"   🔍 分析图片 {idx+1}/{len(image_paths)}...")
                                # 使用_extract_keywords_for_caption方法，生成简洁的caption
                                new_caption = self._extract_keywords_for_caption(img_path, text_content=text)
                                new_captions.append({
                                    "position": f"image_{idx}",
                                    "caption": new_caption
                                })
                                if isinstance(new_caption, dict):
                                    print(f"      📸 图片{idx}: {new_caption.get('zh', '')} ({new_caption.get('en', '')})")
                                else:
                                    print(f"      📸 图片{idx}: {new_caption}")
                            else:
                                print(f"      ⚠️  图片{idx}不存在，跳过")
                        
                        if not new_captions:
                            print(f"   ⚠️  无法生成caption，跳过")
                            continue
                        
                        print(f"   ✅ 生成了 {len(new_captions)} 个新caption")
                        
                        # Step 2: 根据新生成的caption和部分上下文重新生成图片
                        print(f"\n   🎨 Step 2: 根据新caption和部分上下文重新生成图片...")
                        
                        # 提取caption的英文版本用于图片生成
                        caption_guidance_list = []
                        for cap_item in new_captions:
                            cap = cap_item.get('caption')
                            if isinstance(cap, dict):
                                caption_en = cap.get('en', '')
                                caption_zh = cap.get('zh', '')
                            else:
                                caption_en = str(cap)
                                caption_zh = str(cap)
                            caption_guidance_list.append({
                                'en': caption_en,
                                'zh': caption_zh
                            })
                        
                        # 重新生成图片（根据新caption和部分上下文）
                        new_image_paths = self._regenerate_images_with_new_captions(
                            html_path=current_html_path,
                            text_content=text,
                            user_profile=profile_text,
                            output_dir=output_dir,
                            new_captions=caption_guidance_list,
                            tags=tags,
                            rag_examples=rag_examples
                        )
                        
                        if new_image_paths:
                            # 替换HTML中的图片
                            new_html_path = self._replace_images_in_html(
                                html_path=current_html_path,
                                new_image_paths=new_image_paths,
                                iteration=iteration
                            )
                            
                            if new_html_path:
                                current_html_path = new_html_path
                                print(f"   ✅ 图片重新生成成功")
                            else:
                                print(f"   ⚠️  HTML更新失败，继续下一次迭代")
                                # 不break，继续下一次迭代
                                continue
                        else:
                            print(f"   ⚠️  图片重新生成失败，继续下一次迭代")
                            # 不break，继续下一次迭代
                            continue
                        
                        # Step 3: 应用新生成的caption（只有图片生成成功才会执行到这里）
                        should_apply = True
                        suggestion_detail = {
                            "text_changes": [],
                            "image_captions": new_captions
                        }
                    
                    else:
                        # 其他iteration，使用默认逻辑
                        print(f"   ⚠️  未知的iteration: {iteration}，跳过")
                        continue
                    
                    # 注意：第三次reflection时，should_apply和suggestion_detail已在上面设置
                    if should_apply:
                        print(f"\n   🔧 应用改进建议...")
                        
                        num_image_captions = len(suggestion_detail.get('image_captions', []))
                        num_text_changes = len(suggestion_detail.get('text_changes', []))
                        
                        print(f"   - 图片caption: {num_image_captions} 条")
                        print(f"   - 文本修改: {num_text_changes} 条")
                        
                        if num_image_captions > 0 or num_text_changes > 0:
                            new_html_path = self._apply_reflection_suggestions(
                                current_html_path,
                                suggestion_detail,
                                iteration
                            )
                            
                            if new_html_path:
                                # 如果启用严格模式，验证新版本的score是否提升
                                if self.reflection_strict_mode:
                                    print(f"\n      🔍 验证改进效果（严格模式）...")
                                    try:
                                        new_eval_result = evaluate_file(
                                            html_path=new_html_path,
                                            evaluator=self.clip_evaluator,
                                            use_combined=True,
                                            verbose=False
                                        )
                                        new_score = new_eval_result.group_score_mean  # 使用mean作为评估指标
                                        score_delta = new_score - groupscore
                                        
                                        print(f"      📊 修改前 (Mean): {groupscore:.4f}")
                                        print(f"      📊 修改后 (Mean): {new_score:.4f}")
                                        print(f"      📊 变化: {score_delta:+.4f}")
                                        
                                        if new_score >= groupscore:
                                            current_html_path = new_html_path
                                            final_version = f"v{iteration+1}"
                                            print(f"      ✅ Score提升或持平，接受修改: {Path(new_html_path).name}")
                                        else:
                                            print(f"      ❌ Score降低，拒绝修改（保留v{iteration}）")
                                            # 保留失败的版本（不删除，便于调试）
                                            # 但不更新current_html_path，继续使用上一版本
                                            # 继续下一次迭代（可能其他建议会更好）
                                    except Exception as e:
                                        print(f"      ⚠️  验证失败: {e}，默认接受修改")
                                        current_html_path = new_html_path
                                        final_version = f"v{iteration+1}"
                                else:
                                    # 非严格模式：直接接受修改
                                    current_html_path = new_html_path
                                    final_version = f"v{iteration+1}"
                                    print(f"   ✅ 已生成改进版本: {Path(new_html_path).name}")
                            else:
                                print(f"   ⚠️  应用建议失败，继续下一次迭代")
                                # 不break，继续下一次迭代（可能下一次会成功）
                                continue
                        else:
                            print(f"   ⚠️  没有具体的修改内容，继续下一次迭代")
                            # 不break，继续下一次迭代（可能下一次会有内容）
                            continue
                    else:
                        # 只有在第3次及以后的迭代中，如果没有修改建议，才继续下一次迭代
                        # 前3次如果没有修改建议，可能是真的没有需要改进的地方
                        if iteration >= 2:
                            print(f"   ℹ️  无修改建议，继续下一次迭代（第{iteration+1}次反思）")
                            continue
                        else:
                            print(f"   ℹ️  无修改建议，停止reflection")
                            break
                    
                except Exception as e:
                    print(f"\n   ❌ Reflection过程出错: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            # 检查最终版本是否已记录在history中（可能最后一次迭代生成了新版本但未评估）
            final_version_recorded = any(
                record['version'] == final_version 
                for record in reflection_history
            )
            
            if not final_version_recorded and current_html_path:
                # 最终版本未记录，需要评估并记录
                print(f"\n📊 评估最终版本 {final_version} 的GroupScore...")
                try:
                    final_eval_result = evaluate_file(
                        html_path=current_html_path,
                        evaluator=self.clip_evaluator,
                        use_combined=True,
                        verbose=False
                    )
                    
                    if final_eval_result:
                        final_groupscore = final_eval_result.group_score_mean
                        print(f"   📈 GroupScore (Mean): {final_groupscore:.4f}")
                        
                        # 添加到history
                        reflection_history.append({
                            "iteration": len(reflection_history) + 1,
                            "version": final_version,
                            "groupscore": final_groupscore,
                            "html_path": current_html_path
                        })
                    else:
                        print(f"   ⚠️  最终版本评估失败")
                except Exception as e:
                    print(f"   ⚠️  评估最终版本时出错: {e}")
            
            # 打印Reflection总结
            print(f"\n{'='*80}")
            print(f"📋 Reflection总结")
            print(f"{'='*80}")
            print(f"总迭代次数: {len(reflection_history)}")
            print(f"最终版本: {final_version}")
            
            if reflection_history:
                print(f"\nGroupScore变化:")
                for record in reflection_history:
                    status = "✅ 达标" if record["groupscore"] >= self.reflection_threshold else "⚠️  未达标"
                    strategy_info = ""
                    if 'strategy' in record:
                        if record['strategy'] == 'image_regeneration':
                            strategy_info = " [图片重建]"
                        elif record['strategy'] == 'image_generation_rescue':
                            strategy_info = " [图片补救]"
                    if record.get('switched_to_best'):
                        strategy_info += " [基于最佳版本]"
                    print(f"  {record['version']}: {record['groupscore']:.4f} {status}{strategy_info}")
            
            print(f"\n最终HTML: {Path(current_html_path).name}")
            print(f"{'='*80}\n")
        
        return {
            "text": text,
            "images": images,
            "links": links,
            "tags": tags,
            "html_post": current_html_path,  # 返回最终版本的HTML路径
            "reflection_history": reflection_history,  # 返回reflection历史
            "final_version": final_version
        }