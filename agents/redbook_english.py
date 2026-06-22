import os
import json
import glob
import requests
import re
from datetime import datetime

# 尝试加载 .env 文件（如果还没有加载）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv 未安装，跳过


class RedBookEnglishConverter:
    """将小红书风格的中文帖子转换为符合国外社交媒体（TikTok/YouTube）风格的英文帖子"""
    
    def __init__(self, generated_dir="generated_it"):
        self.generated_dir = generated_dir
        
        # 使用环境变量中的 API 配置
        self.chat_api_key = os.getenv("CHAT_API_KEY")
        self.chat_base_url = os.getenv("CHAT_BASE_URL")
        self.chat_model = os.getenv("CHAT_MODEL")
        
        # 联网搜索模型（用于获取真实链接）
        self.search_api_key = os.getenv("SEARCH_API_KEY")
        self.search_base_url = os.getenv("SEARCH_BASE_URL", "https://yunwu.ai/v1")
        self.search_model = os.getenv("SEARCH_MODEL", "gpt-5-all")
    
    def get_available_posts(self):
        """获取所有已生成的帖子目录"""
        if not os.path.exists(self.generated_dir):
            return []
        
        posts = []
        for item in os.listdir(self.generated_dir):
            item_path = os.path.join(self.generated_dir, item)
            if os.path.isdir(item_path):
                # 检查是否有 final_results.json
                result_file = os.path.join(item_path, "final_results.json")
                if os.path.exists(result_file):
                    posts.append(item)
        
        return sorted(posts)
    
    def load_post_data(self, user_dir):
        """加载帖子数据"""
        user_path = os.path.join(self.generated_dir, user_dir)
        result_file = os.path.join(user_path, "final_results.json")
        
        if not os.path.exists(result_file):
            return None
        
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def _extract_from_html(self, html_path, fallback_post_data):
        """
        从HTML中提取改进后的文本内容
        
        Args:
            html_path: HTML文件路径
            fallback_post_data: 如果HTML解析失败，使用的fallback数据
            
        Returns:
            (text, tags, images, links) tuple
        """
        try:
            if not os.path.exists(html_path):
                print(f"   ⚠️  HTML文件不存在，使用fallback数据: {html_path}")
                return (
                    fallback_post_data.get("text", ""),
                    fallback_post_data.get("tags", []),
                    fallback_post_data.get("images", []),
                    fallback_post_data.get("links", [])
                )
            
            from bs4 import BeautifulSoup
            
            with open(html_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            # 提取标签
            tags = []
            tag_elements = soup.find_all('span', class_='tag')
            for tag_elem in tag_elements:
                tag_text = tag_elem.get_text(strip=True)
                # 去除 # 符号
                tag_text = tag_text.replace('#', '').strip()
                if tag_text:
                    tags.append(tag_text)
            
            # 提取文本内容
            content_div = soup.find('div', class_='post-content')
            if not content_div:
                print(f"   ⚠️  未找到post-content，使用fallback数据")
                return (
                    fallback_post_data.get("text", ""),
                    tags or fallback_post_data.get("tags", []),
                    fallback_post_data.get("images", []),
                    fallback_post_data.get("links", [])
                )
            
            # 提取所有段落文本（排除链接卡片）
            paragraphs = []
            for elem in content_div.find_all(['p', 'h3']):
                text = elem.get_text(strip=True)
                if text:
                    paragraphs.append(text)
            
            chinese_text = '\n\n'.join(paragraphs)
            
            # 从HTML中提取图片路径（这样可以获取重新生成的图片）
            images = []
            html_dir = os.path.dirname(html_path)
            for img_div in soup.find_all('div', class_='post-image'):
                img_tag = img_div.find('img')
                if img_tag and img_tag.get('src'):
                    img_src = img_tag['src']
                    # 如果是相对路径，转换为绝对路径
                    if not os.path.isabs(img_src):
                        img_path = os.path.join(html_dir, img_src)
                    else:
                        img_path = img_src
                    # 规范化路径
                    img_path = os.path.normpath(img_path)
                    images.append(img_path)
            
            # 如果HTML中没有图片，尝试从fallback获取
            if not images:
                print(f"   ⚠️  HTML中未找到图片，使用fallback数据")
                images = fallback_post_data.get("images", [])
            
            # links仍从fallback数据获取（HTML中只是引用，实际数据在JSON中）
            links = fallback_post_data.get("links", [])
            
            print(f"   ✅ 从HTML提取: {len(chinese_text)} 字符, {len(tags)} 标签, {len(images)} 图片")
            
            return (chinese_text, tags, images, links)
            
        except Exception as e:
            print(f"   ⚠️  HTML解析失败: {e}，使用fallback数据")
            return (
                fallback_post_data.get("text", ""),
                fallback_post_data.get("tags", []),
                fallback_post_data.get("images", []),
                fallback_post_data.get("links", [])
            )
    
    def convert_to_english(self, chinese_text, tags, links):
        """将中文文案、标签和链接标题转换为符合国外社交媒体风格的英文"""
        
        # 构建链接信息
        links_info = ""
        if links:
            links_info = "\n**Original Link Titles (need translation):**\n"
            for i, link in enumerate(links):
                links_info += f"{i+1}. {link.get('title', '')}\n"
        
        prompt = f"""
You are a professional social media content creator who specializes in creating engaging content for TikTok, YouTube, and Instagram. Your task is to transform Chinese social media content into natural, platform-appropriate English content.

**Original Chinese Content:**
{chinese_text}

**Original Tags:**
{', '.join(tags) if tags else 'None'}
{links_info}

**Conversion Requirements:**

1. **Language Style - Match TikTok/YouTube/Instagram Culture:**
   - Use casual, conversational English like talking to friends
   - Popular phrases: "Guys!", "Besties!", "Y'all", "No cap", "Literally", "Fr fr" (for real), "Ngl" (not gonna lie)
   - Enthusiastic expressions: "OMG!", "This is insane!", "I'm obsessed!", "You NEED this!", "Game changer!"
   - For recommendations: "Highly recommend", "10/10 would recommend", "You're missing out", "Trust me on this"
   - For warnings: "Skip this", "Save your money", "Major red flag"
   - Use "lowkey/highkey" for emphasis
   - Use emojis naturally but don't overdo it

2. **Content Adaptation (NOT Direct Translation):**
   - Keep the same message and tone but adapt cultural references
   - If mentioning Chinese platforms (小红书/B站/抖音), convert to equivalent:
     * 小红书 → Instagram/Pinterest
     * B站 → YouTube
     * 抖音 → TikTok
     * 知乎 → Reddit/Quora
     * 微博 → Twitter/X
   - Adapt Chinese slang to English internet slang
   - Keep specific place names, product names, and prices as-is (or add USD conversion if relevant)

3. **Structure:**
   - Start with an engaging hook (like TikTok/YouTube intros)
   - Keep paragraphs short and punchy
   - Use line breaks for emphasis
   - End with a call-to-action or question to boost engagement

4. **Tags:**
   - Convert tags to English hashtags
   - Make them TikTok/Instagram-friendly
   - Use popular English hashtags format

5. **Link Titles:**
   - Convert link titles to natural, engaging English
   - Make them clickable and appealing
   - Keep the same topic/theme

**Output Format:**
Return ONLY a JSON object (no markdown formatting):
{{
  "english_text": "The converted English content here...",
  "english_tags": ["tag1", "tag2", "tag3", "tag4"],
  "english_link_titles": ["English title 1", "English title 2"]
}}

IMPORTANT: 
- Return PURE JSON only, no ```json or ``` wrappers
- Make it sound natural and native, NOT like a translation
- Keep the enthusiasm and energy of the original
- The number of english_link_titles should match the number of original link titles
"""
        
        try:
            resp = requests.post(
                f"{self.chat_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.chat_api_key}"},
                json={
                    "model": self.chat_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,  # 稍高一点，让内容更有创意
                    "top_p": 0.9,
                }
            )
            
            result = resp.json()["choices"][0]["message"]["content"].strip()
            
            # 清洗可能的 markdown 代码块
            if result.startswith("```json"):
                result = result[7:]
            elif result.startswith("```"):
                result = result[3:]
            
            if result.endswith("```"):
                result = result[:-3]
            
            result = result.strip()
            
            # 解析 JSON
            converted = json.loads(result)
            return (
                converted.get("english_text", ""), 
                converted.get("english_tags", []),
                converted.get("english_link_titles", [])
            )
        
        except Exception as e:
            print(f"❌ 转换失败: {e}")
            return None, None, None
    
    def _generate_international_search_url(self, platform, keyword):
        """根据平台和关键词生成真实的搜索链接（支持国外和中文平台）"""
        import urllib.parse
        kw_encoded = urllib.parse.quote(keyword)
        p = platform.lower()
        
        # 国外平台
        if "instagram" in p:
            # Instagram 使用 tag 搜索
            return f"https://www.instagram.com/explore/tags/{kw_encoded}/"
        elif "youtube" in p:
            return f"https://www.youtube.com/results?search_query={kw_encoded}"
        elif "tiktok" in p:
            return f"https://www.tiktok.com/search?q={kw_encoded}"
        elif "reddit" in p:
            return f"https://www.reddit.com/search/?q={kw_encoded}"
        elif "twitter" in p or p == "x":
            return f"https://twitter.com/search?q={kw_encoded}"
        elif "pinterest" in p:
            return f"https://www.pinterest.com/search/pins/?q={kw_encoded}"
        
        # 中文平台（作为备选）
        elif "小红书" in p or "xiaohongshu" in p:
            return f"https://www.xiaohongshu.com/search_result?keyword={kw_encoded}"
        elif "b站" in p or "bilibili" in p:
            return f"https://search.bilibili.com/all?keyword={kw_encoded}"
        elif "知乎" in p or "zhihu" in p:
            return f"https://www.zhihu.com/search?type=content&q={kw_encoded}"
        elif "抖音" in p or "douyin" in p:
            return f"https://www.douyin.com/search/{kw_encoded}"
        elif "微博" in p or "weibo" in p:
            return f"https://s.weibo.com/weibo?q={kw_encoded}"
        
        else:
            # 默认使用 Google 搜索
            return f"https://www.google.com/search?q={kw_encoded}"
    
    def _is_homepage_url(self, url):
        """检测是否为首页链接（需要过滤掉）"""
        if not url or url == "#":
            return True
        
        # 提取路径部分
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.strip('/')
        query = parsed.query
        
        # 如果没有路径或查询参数，可能是首页
        if not path and not query:
            return True
        
        # 如果只有根路径且没有搜索参数
        homepage_patterns = [
            r'^/?$',
            r'^/?index\.(html?|php)$',
            r'^/?home$',
        ]
        
        for pattern in homepage_patterns:
            if re.match(pattern, path):
                return True
        
        return False
    
    def _search_real_international_links(self, links_info, text_content):
        """
        使用联网搜索获取真实的国外平台帖子链接
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
        for i, info in enumerate(links_info):
            title = info.get('english_title', '')
            platform = info.get('target_platform', '')
            keyword = info.get('search_keyword', '')
            links_desc.append(f"{i+1}. Platform: {platform}, Keyword: {keyword}, Expected: {title}")
        
        search_prompt = f"""Based on the English post content and recommended topics, search and return 2 **real post/video links** from ANY platform.

Post content (first 600 chars):
{text_content[:600]}

Recommended directions:
{chr(10).join(links_desc)}

Return JSON format:
[
  {{
    "title": "Actual post/video title (in English)",
    "platform": "Platform name (can be ANY: Instagram/YouTube/TikTok/Reddit/Twitter/小红书/B站/知乎/微博/抖音/etc.)",
    "url": "Real post/video URL (must be specific post, not homepage or search page)",
    "search_keyword": "Precise search keyword (only if really can't find any specific post link)"
  }}
]

**CRITICAL Requirements:**
1. **URL MUST be a SPECIFIC post/video/discussion link, NOT homepage or search page**
2. **Priority order:**
   - First try: Instagram, YouTube, TikTok, Reddit, Twitter (international platforms)
   - Second try: 小红书, B站, 知乎, 微博, 抖音 (Chinese platforms are OK if they have specific posts)
   - Last resort: Other platforms are also acceptable
3. **Title should be in English** (translate if the actual post is in Chinese)
4. **Only use search_keyword if you absolutely cannot find ANY specific post link**
5. Prefer ANY specific post URL over search pages"""

        try:
            print(f"🔍 Searching for real international links using {self.search_model}...")
            response = requests.post(
                f"{self.search_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.search_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.search_model,
                    "web_search_options": {},  # Enable web search
                    "messages": [{"role": "user", "content": search_prompt}],
                    "temperature": 0.7
                },
                timeout=90
            )
            
            if response.status_code != 200:
                print(f"⚠️ Search API error {response.status_code}")
                return None
            
            resp_json = response.json()
            
            if "choices" not in resp_json:
                print(f"⚠️ Unexpected response format")
                return None
            
            content = resp_json["choices"][0]["message"]["content"]
            
            # Clean content
            content = re.sub(r'^>.*?$', '', content, flags=re.MULTILINE)
            content = re.sub(r'\*\*\[.*?\]\(.*?\)\*\*\s*·\s*\*.*?\*', '', content)
            content = re.sub(r'\[.*?\]\(.*?\)', '', content)
            content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'^```\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'\n{2,}', '\n', content).strip()
            
            # Extract JSON array
            json_content = None
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
            
            if not json_content:
                print(f"⚠️ Could not extract valid JSON array")
                return None
            
            # Parse JSON
            try:
                search_results = json.loads(json_content)
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parse error: {e}")
                return None
            
            # Validate and extract valid links (accept ANY platform)
            valid_links = []
            if isinstance(search_results, list):
                for result in search_results:
                    if not isinstance(result, dict):
                        continue
                    
                    # Prioritize real URLs (from ANY platform)
                    if result.get("url") and result.get("title"):
                        url = result.get("url", "")
                        # Validate URL format
                        if not (url.startswith("http://") or url.startswith("https://")):
                            continue
                        
                        # Filter homepage links (critical: no search pages!)
                        if self._is_homepage_url(url):
                            print(f"⚠️ Filtered homepage link: {url[:60]}...")
                            continue
                        
                        # Check if it's a search page (additional check)
                        if any(indicator in url.lower() for indicator in ['/search?', '/search/', 'search_query=', 'search_result']):
                            print(f"⚠️ Filtered search page: {url[:60]}...")
                            continue
                        
                        platform = result.get("platform", "Web")
                        valid_links.append({
                            "title": result.get("title", ""),
                            "platform": platform,
                            "url": url
                        })
                        # 区分国外和中文平台
                        platform_type = "🌍 International" if any(p in platform.lower() for p in ['instagram', 'youtube', 'tiktok', 'reddit', 'twitter']) else "🇨🇳 Chinese"
                        print(f"✅ Found real link [{platform_type}]: {result.get('title', '')[:40]}... -> {url[:60]}...")
                    
                    # Fallback: use search keyword (only if really necessary)
                    elif result.get("search_keyword") and result.get("platform"):
                        keyword = result.get("search_keyword", "")
                        platform = result.get("platform", "")
                        # Try to generate appropriate search URL based on platform
                        search_url = self._generate_international_search_url(platform, keyword)
                        
                        valid_links.append({
                            "title": result.get("title", f"{platform} search: {keyword}"),
                            "platform": platform,
                            "url": search_url
                        })
                        print(f"⚠️ Using search link (last resort): {result.get('title', '')[:40]}... | Keyword: {keyword}")
            
            if valid_links:
                print(f"✅ Successfully retrieved {len(valid_links)} links")
                return valid_links
            else:
                print(f"⚠️ No valid links found")
                return None
        
        except Exception as e:
            print(f"⚠️ Search exception: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def adapt_links_to_english(self, links, english_link_titles):
        """将国内平台链接转换为国外平台建议，使用英文标题（优先获取真实链接）"""
        if not links:
            return []
        
        # 平台映射
        platform_map = {
            "小红书": "Instagram",
            "b站": "YouTube",
            "bilibili": "YouTube",
            "知乎": "Reddit",
            "抖音": "TikTok",
            "微博": "Twitter"
        }
        
        # 准备链接信息用于联网搜索
        links_info = []
        for i, link in enumerate(links):
            original_platform = link.get("platform", "")
            original_url = link.get("url", "")
            
            # 使用转换后的英文标题
            english_title = link.get("title", "")
            if english_link_titles and i < len(english_link_titles):
                english_title = english_link_titles[i]
            
            # 从原始 URL 中提取关键词
            search_keyword = ""
            if "keyword=" in original_url:
                try:
                    import urllib.parse
                    parsed = urllib.parse.urlparse(original_url)
                    params = urllib.parse.parse_qs(parsed.query)
                    if 'keyword' in params:
                        search_keyword = params['keyword'][0]
                    elif 'q' in params:
                        search_keyword = params['q'][0]
                except:
                    pass
            
            # 如果没有提取到，使用英文标题作为关键词
            if not search_keyword:
                search_keyword = english_title
            
            # 转换平台名
            target_platform = original_platform
            for cn, en in platform_map.items():
                if cn in original_platform.lower():
                    target_platform = en
                    break
            
            links_info.append({
                "english_title": english_title,
                "target_platform": target_platform,
                "search_keyword": search_keyword
            })
        
        # 尝试获取真实链接
        print("🔍 Attempting to retrieve real international post links...")
        real_links = self._search_real_international_links(links_info, "")
        
        if real_links:
            return real_links
        
        # 降级：生成搜索链接
        print("⚠️ Web search failed, using search URLs as fallback")
        adapted_links = []
        for info in links_info:
            search_url = self._generate_international_search_url(
                info['target_platform'], 
                info['search_keyword']
            )
            
            adapted_links.append({
                "title": info['english_title'],
                "platform": info['target_platform'],
                "url": search_url,
                "search_keyword": info['search_keyword']
            })
        
        return adapted_links
    
    def generate_english_html(self, english_text, english_tags, image_paths, adapted_links, output_path):
        """生成英文版 HTML"""
        
        # 按双换行符分割段落
        raw_paragraphs = [p for p in english_text.split('\n\n') if p.strip()]
        
        # 渲染 markdown
        processed_paragraphs = []
        for p in raw_paragraphs:
            # 处理加粗
            rendered = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #333; font-weight: 700;">\1</strong>', p)
            processed_paragraphs.append(rendered)
        
        paragraphs = processed_paragraphs
        
        # 组装内容
        html_parts = []
        insertions = []
        
        # 插入图片
        for i, img_path in enumerate(image_paths):
            insertions.append({"type": "image", "content": os.path.basename(img_path), "index": i})
        
        # 插入链接
        for link in adapted_links:
            insertions.append({"type": "link", "content": link})
        
        num_paras = len(paragraphs)
        
        if num_paras == 0:
            html_parts.append(english_text)
        else:
            num_inserts = len(insertions)
            if num_inserts > 0:
                step = max(1, num_paras // (num_inserts + 1))
                current_insert_idx = 0
                
                for i, para in enumerate(paragraphs):
                    if para.startswith('<h3'):
                        html_parts.append(para)
                    else:
                        html_parts.append(f"<p>{para}</p>")
                    
                    if current_insert_idx < num_inserts:
                        if (i + 1) % step == 0 or i == num_paras - 1:
                            item = insertions[current_insert_idx]
                            if item["type"] == "image":
                                html_parts.append(self._create_image_tag(item["content"], item["index"]))
                            elif item["type"] == "link":
                                html_parts.append(self._create_link_tag(item["content"]))
                            current_insert_idx += 1
                            
                            if i == num_paras - 1:
                                while current_insert_idx < num_inserts:
                                    item = insertions[current_insert_idx]
                                    if item["type"] == "image":
                                        html_parts.append(self._create_image_tag(item["content"], item["index"]))
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
        
        # 生成标签 HTML
        tags_html = "".join([f'<span class="tag"># {tag}</span>' for tag in english_tags])
        
        # HTML 模板
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Social Media Post - English Version</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.75; max-width: 800px; margin: 0 auto; padding: 15px; background: #f5f5f5; color: #333; }}
        .post-container {{ background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin: 10px 0; }}
        
        .post-content {{ font-size: 17px; color: #2c3e50; letter-spacing: 0.02em; }}
        .post-content p {{ margin: 1em 0; text-align: left; }}
        .post-content strong {{ color: #000; font-weight: 700; background: linear-gradient(to bottom, transparent 60%, #fffbe6 60%); }}
        .post-content h3 {{ font-size: 1.2em; margin-top: 1.5em; margin-bottom: 0.5em; color: #1a1a1a; }}
        
        .post-image {{ margin: 20px -25px; width: calc(100% + 50px); text-align: center; }}
        .post-image img {{ width: 100%; display: block; }}
        .image-caption {{ color: #999; font-size: 13px; margin-top: 8px; font-style: italic; padding: 0 25px; }}
        
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
        .link-card.platform-instagram {{ border-left: 5px solid #E4405F; }}
        .link-card.platform-youtube {{ border-left: 5px solid #FF0000; }}
        .link-card.platform-tiktok {{ border-left: 5px solid #000000; }}
        .link-card.platform-reddit {{ border-left: 5px solid #FF4500; }}
        .link-card.platform-twitter {{ border-left: 5px solid #1DA1F2; }}
        .link-card.platform-xiaohongshu {{ border-left: 5px solid #ff2442; }}
        .link-card.platform-bilibili {{ border-left: 5px solid #23ade5; }}
        .link-card.platform-zhihu {{ border-left: 5px solid #0084ff; }}
        .link-card.platform-douyin {{ border-left: 5px solid #1c1e21; }}
        .link-card.platform-weibo {{ border-left: 5px solid #ea5d5c; }}
        
        .link-info {{ flex: 1; }}
        .link-platform-tag {{ 
            font-size: 12px; font-weight: bold; margin-bottom: 4px; display: inline-block; padding: 2px 6px; border-radius: 4px; color: white;
        }}
        .tag-instagram {{ background: #E4405F; }}
        .tag-youtube {{ background: #FF0000; }}
        .tag-tiktok {{ background: #000; }}
        .tag-reddit {{ background: #FF4500; }}
        .tag-twitter {{ background: #1DA1F2; }}
        .tag-xiaohongshu {{ background: #ff2442; }}
        .tag-bilibili {{ background: #23ade5; }}
        .tag-zhihu {{ background: #0084ff; }}
        .tag-douyin {{ background: #000; }}
        .tag-weibo {{ background: #ea5d5c; }}
        
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
                <h3>AI Content Creator</h3>
                <div class="post-time">{datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
            </div>
        </div>
        <div class="post-tags">
            {tags_html}
        </div>
        <div class="post-content">{html_content}</div>
        <div style="margin-top:30px; border-top:1px solid #eee; padding-top:15px; color:#ccc; font-size:12px; text-align:center;">
            Generated by AI • English Version • {len(image_paths)} Images
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        return output_path
    
    def _create_image_tag(self, image_filename, index):
        return f'<div class="post-image"><img src="{image_filename}"><div class="image-caption">Image {index + 1}</div></div>'
    
    def _create_link_tag(self, link_data):
        title = link_data.get('title', 'Related Content')
        platform = link_data.get('platform', 'Web').strip()
        url = link_data.get('url', '#')
        
        css_class = "platform-other"
        tag_class = "tag-other"
        icon = "🔗"
        
        p = platform.lower()
        # 国外平台
        if "instagram" in p:
            css_class = "platform-instagram"
            tag_class = "tag-instagram"
            icon = "📷"
        elif "youtube" in p:
            css_class = "platform-youtube"
            tag_class = "tag-youtube"
            icon = "📺"
        elif "tiktok" in p:
            css_class = "platform-tiktok"
            tag_class = "tag-tiktok"
            icon = "🎵"
        elif "reddit" in p:
            css_class = "platform-reddit"
            tag_class = "tag-reddit"
            icon = "💬"
        elif "twitter" in p or "x" == p:
            css_class = "platform-twitter"
            tag_class = "tag-twitter"
            icon = "🐦"
        # 中文平台（作为备选）
        elif "小红书" in p or "xiaohongshu" in p:
            css_class = "platform-xiaohongshu"
            tag_class = "tag-xiaohongshu"
            icon = "📕"
        elif "b站" in p or "bilibili" in p:
            css_class = "platform-bilibili"
            tag_class = "tag-bilibili"
            icon = "📺"
        elif "知乎" in p or "zhihu" in p:
            css_class = "platform-zhihu"
            tag_class = "tag-zhihu"
            icon = "❓"
        elif "抖音" in p or "douyin" in p:
            css_class = "platform-douyin"
            tag_class = "tag-douyin"
            icon = "🎵"
        elif "微博" in p or "weibo" in p:
            css_class = "platform-weibo"
            tag_class = "tag-weibo"
            icon = "👁️"
        
        return f'''
<a href="{url}" class="link-card {css_class}" target="_blank">
    <div class="link-icon">{icon}</div>
    <div class="link-info">
        <span class="link-platform-tag {tag_class}">{platform}</span>
        <div class="link-title">{title}</div>
        <div class="link-action">Check it out on {platform} &gt;</div>
    </div>
</a>
'''
    
    def convert_post(self, user_dir):
        """转换单个帖子"""
        print(f"\n{'='*50}")
        print(f"Converting: {user_dir}")
        print(f"{'='*50}")
        
        # 加载原始数据
        data = self.load_post_data(user_dir)
        if not data:
            print(f"❌ 无法加载帖子数据")
            return None
        
        post_data = data.get("personalized_post", {})
        
        # ===== 使用最终版本的HTML（已经是最优版本） =====
        # html_post 字段已经是 current_html_path，即最终版本
        best_html_path = post_data.get("html_post", "")
        final_version = post_data.get("final_version", "v0")
        reflection_history = post_data.get("reflection_history", [])
        
        if reflection_history:
            print(f"📊 Reflection历史:")
            for record in reflection_history:
                is_final = (record.get('version') == final_version)
                marker = "⭐ (最终版本)" if is_final else "  "
                strategy = ""
                if 'strategy' in record:
                    if record['strategy'] == 'image_regeneration':
                        strategy = " [图片重建]"
                    elif record['strategy'] == 'image_generation_rescue':
                        strategy = " [图片补救]"
                if record.get('switched_to_best'):
                    strategy += " [基于最佳版本]"
                print(f"   {marker} {record['version']}: GroupScore = {record['groupscore']:.4f}{strategy}")
            print(f"✅ 使用最终版本: {final_version} (html_post已指向最终版本)")
        else:
            print(f"ℹ️  无Reflection历史，使用初始版本: {final_version}")
        
        # 从选中的HTML中提取内容（而不是直接使用原始text）
        chinese_text, chinese_tags, images, links = self._extract_from_html(best_html_path, post_data)
        
        if not chinese_text:
            print(f"❌ 没有文本内容")
            return None
        
        print(f"📝 原始文本长度: {len(chinese_text)} 字符")
        print(f"🏷️  原始标签: {chinese_tags}")
        if links:
            print(f"🔗 原始链接: {[l.get('title', '') for l in links]}")
        
        # 转换为英文
        print("🔄 Converting to English...")
        english_text, english_tags, english_link_titles = self.convert_to_english(chinese_text, chinese_tags, links)
        
        if not english_text:
            print(f"❌ 转换失败")
            return None
        
        print(f"✅ 英文文本长度: {len(english_text)} 字符")
        print(f"🏷️  英文标签: {english_tags}")
        if english_link_titles:
            print(f"🔗 英文链接标题: {english_link_titles}")
        
        # 适配链接（使用英文标题）
        adapted_links = self.adapt_links_to_english(links, english_link_titles)
        
        # 生成英文版 HTML
        user_path = os.path.join(self.generated_dir, user_dir)
        output_html = os.path.join(user_path, "image_text_english.html")
        
        print("📄 Generating English HTML...")
        self.generate_english_html(english_text, english_tags, images, adapted_links, output_html)
        
        # 保存英文数据
        english_data = {
            "text": english_text,
            "tags": english_tags,
            "images": images,
            "links": adapted_links,
            "html_path": output_html
        }
        
        english_json = os.path.join(user_path, "english_post.json")
        with open(english_json, 'w', encoding='utf-8') as f:
            json.dump(english_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 完成！查看: {output_html}")
        
        return english_data
    
    def convert_all(self):
        """转换所有帖子"""
        posts = self.get_available_posts()
        
        if not posts:
            print("❌ 没有找到任何已生成的帖子")
            return
        
        print(f"\n找到 {len(posts)} 个帖子")
        print(f"{'='*50}")
        
        success_count = 0
        for post_dir in posts:
            try:
                result = self.convert_post(post_dir)
                if result:
                    success_count += 1
            except Exception as e:
                print(f"❌ 转换 {post_dir} 时出错: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*50}")
        print(f"🎉 完成！成功转换 {success_count}/{len(posts)} 个帖子")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='将小红书帖子转换为英文版（TikTok/YouTube风格）')
    parser.add_argument('--user', type=str, help='指定要转换的用户目录名')
    parser.add_argument('--all', action='store_true', help='转换所有帖子')
    
    args = parser.parse_args()
    
    # 设置环境变量（如果需要）
    if not os.getenv("CHAT_API_KEY"):
        os.environ.update({
            "CHAT_API_KEY": "sk-dVaSXmTEMBh0Gygx49ResSvaONvErml5QV8McBAGkbPmX2mG",
            "CHAT_BASE_URL": "https://yunwu.ai/v1",
            "CHAT_MODEL": "gpt-4o",
        })
    
    converter = RedBookEnglishConverter()
    
    if args.all:
        converter.convert_all()
    elif args.user:
        converter.convert_post(args.user)
    else:
        # 交互式选择
        posts = converter.get_available_posts()
        if not posts:
            print("❌ 没有找到任何已生成的帖子")
            return
        
        print(f"\n{'='*50}")
        print("可用的帖子:")
        print(f"{'='*50}")
        for idx, post in enumerate(posts):
            print(f"[{idx}] {post}")
        print(f"{'='*50}")
        
        user_input = input("\n请输入要转换的帖子序号（多个用空格分隔，或输入 'all' 转换所有）: ").strip()
        
        if user_input.lower() == 'all':
            converter.convert_all()
        else:
            # 解析输入的序号列表
            selected_posts = []
            for item in user_input.split():
                item = item.strip()
                if item.isdigit():
                    idx = int(item)
                    if 0 <= idx < len(posts):
                        if posts[idx] not in selected_posts:
                            selected_posts.append(posts[idx])
                    else:
                        print(f"⚠️ 警告: 序号 [{idx}] 超出范围 (0-{len(posts)-1})")
                else:
                    print(f"⚠️ 警告: 无效的输入 '{item}'")
            
            if not selected_posts:
                print("❌ 未选择有效的帖子")
                return
            
            print(f"\n✅ 即将转换 {len(selected_posts)} 个帖子:")
            for post in selected_posts:
                print(f"  {post}")
            print(f"{'='*50}\n")
            
            # 批量转换
            success_count = 0
            for post_dir in selected_posts:
                try:
                    result = converter.convert_post(post_dir)
                    if result:
                        success_count += 1
                except Exception as e:
                    print(f"❌ 转换 {post_dir} 时出错: {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"\n{'='*50}")
            print(f"🎉 完成！成功转换 {success_count}/{len(selected_posts)} 个帖子")


if __name__ == "__main__":
    main()

