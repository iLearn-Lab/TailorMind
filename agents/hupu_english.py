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


class HupuEnglishConverter:
    """将虎扑风格的中文讨论帖转换为符合国外论坛（Reddit/Twitter）风格的英文帖"""
    
    def __init__(self, generated_dir="generated_posts"):
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
    
    def _extract_from_html(self, html_path):
        """
        从HTML中提取改进后的文本内容和链接
        
        Args:
            html_path: HTML文件路径
            
        Returns:
            (text, links) tuple
        """
        try:
            if not os.path.exists(html_path):
                print(f"   ⚠️  HTML文件不存在: {html_path}")
                return "", []
            
            from bs4 import BeautifulSoup
            
            with open(html_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            # 提取文本内容
            content_div = soup.find('div', class_='post-content')
            if not content_div:
                print(f"   ⚠️  未找到post-content")
                return "", []
            
            # 提取所有段落文本（排除链接卡片）
            paragraphs = []
            for elem in content_div.find_all('p'):
                text = elem.get_text(strip=True)
                if text:
                    paragraphs.append(text)
            
            chinese_text = '\n\n'.join(paragraphs)
            
            # 提取链接
            links = []
            link_cards = content_div.find_all('a', class_='link-card')
            for link_card in link_cards:
                title_div = link_card.find('div', class_='link-title')
                platform_tag = link_card.find('span', class_='link-platform-tag')
                
                if title_div:
                    link_title = title_div.get_text(strip=True)
                    link_url = link_card.get('href', '')
                    link_platform = platform_tag.get_text(strip=True) if platform_tag else '网页'
                    
                    links.append({
                        'title': link_title,
                        'url': link_url,
                        'platform': link_platform
                    })
            
            print(f"   ✅ 从HTML提取: {len(chinese_text)} 字符, {len(links)} 个链接")
            return chinese_text, links
            
        except Exception as e:
            print(f"   ❌ HTML解析失败: {e}")
            import traceback
            traceback.print_exc()
            return "", []
    
    def convert_to_english(self, chinese_text, hot_topics, links):
        """将中文讨论帖、热点和链接标题转换为符合国外论坛风格的英文
        
        核心策略：不是机翻，而是文化适配
        - 虎扑风格 → Reddit/Twitter 论坛风格
        - 保持讨论的直接性和观点性
        - 适配文化梗和表达方式
        """
        
        # 构建热点信息
        topics_info = ""
        if hot_topics and isinstance(hot_topics, list):
            topics_info = "\n**Original Hot Topics (context):**\n"
            for i, topic in enumerate(hot_topics):
                if isinstance(topic, dict):
                    topics_info += f"{i+1}. {topic.get('topic', '')}\n"
        
        # 构建链接信息
        links_info = ""
        if links:
            links_info = "\n**Original Link Titles (need translation):**\n"
            for i, link in enumerate(links):
                links_info += f"{i+1}. {link.get('title', '')}\n"
        
        prompt = f"""
You are a professional community manager who specializes in adapting Chinese sports/gaming forum content (like Hupu) into natural English forum discussions (like Reddit r/nba, Twitter sports threads).

**Original Chinese Discussion Post:**
{chinese_text}
{topics_info}
{links_info}

**Conversion Requirements:**

1. **Forum Discussion Style - Match Reddit/Twitter Culture:**
   - Write like a knowledgeable fan sharing opinions on a forum
   - Use casual but intelligent tone (think Reddit r/nba or sports Twitter)
   - Common phrases:
     * Opening: "Real talk", "Unpopular opinion but", "Let's be honest", "Can we talk about", "Hot take"
     * Agreement: "Facts", "This right here", "100%", "Exactly", "This is it"
     * Disagreement: "Cap", "Nah", "Hard disagree", "Miss me with that", "Not buying it"
     * Analysis: "The thing is", "Here's the deal", "Look at it this way", "Let me break it down"
     * Closing: "Thoughts?", "Change my mind", "Am I wrong?", "What y'all think?"
   - Use abbreviations naturally: "fr" (for real), "ngl" (not gonna lie), "imo/imho"
   - Sports/Gaming slang is encouraged when relevant

2. **Content Adaptation (Cultural Translation, NOT Word-by-Word):**
   - Adapt "虎扑JRs" → "y'all" / "folks" / "the community"
   - Convert Chinese internet slang to English equivalents:
     * "绷不住" → "can't even", "I'm done", "lmao"
     * "理性讨论" → "Real talk" / "Let's be honest"
     * "蚌埠住了" → "I'm dying", "lmao", "can't handle this"
     * "懂的都懂" → "if you know you know" / "IYKYK"
   - If discussing Chinese platforms, convert to global equivalents:
     * 微博 → Twitter/X
     * 知乎 → Reddit/Quora
     * B站 → YouTube
     * 虎扑 → r/nba or sports forum
   - Keep specific names (players, teams, games, events) as-is
   - For Chinese-specific events/people, add brief context if needed

3. **Structure:**
   - Keep the same logical flow and argument structure
   - Maintain paragraph breaks for readability
   - Preserve the tone (analytical, casual, aggressive, etc.)
   - Keep the discussion format (not a social media "post" but a forum "thread")

4. **Link Titles:**
   - Convert link titles to natural, forum-appropriate English
   - Make them informative and clickable
   - Keep the same topic/theme
   - Use forum-style formatting (like Reddit post titles)

**Output Format:**
Return ONLY a JSON object (no markdown formatting):
{{
  "english_text": "The converted English discussion post here...",
  "english_link_titles": ["English title 1", "English title 2"]
}}

**CRITICAL RULES:**
- Return PURE JSON only, no ```json or ``` wrappers
- Make it sound like a native English-speaking sports/gaming fan wrote it, NOT a translation
- Preserve the original's analytical depth or casual tone
- The number of english_link_titles should match the number of original link titles
- Don't add hashtags or social media elements - this is a FORUM discussion
- Keep it authentic to the original tone (don't make it too "hype" if the original is analytical)
"""
        
        try:
            resp = requests.post(
                f"{self.chat_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.chat_api_key}"},
                json={
                    "model": self.chat_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.75,  # 适中的温度，保持自然但不过度创意
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
                converted.get("english_link_titles", [])
            )
        
        except Exception as e:
            print(f"❌ 转换失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _generate_international_search_url(self, platform, keyword):
        """根据平台和关键词生成真实的搜索链接（支持国外和中文平台）"""
        import urllib.parse
        kw_encoded = urllib.parse.quote(keyword)
        p = platform.lower()
        
        # 国外平台
        if "twitter" in p or p == "x":
            return f"https://twitter.com/search?q={kw_encoded}"
        elif "reddit" in p:
            return f"https://www.reddit.com/search/?q={kw_encoded}"
        elif "youtube" in p:
            return f"https://www.youtube.com/results?search_query={kw_encoded}"
        elif "quora" in p:
            return f"https://www.quora.com/search?q={kw_encoded}"
        elif "tiktok" in p:
            return f"https://www.tiktok.com/search?q={kw_encoded}"
        
        # 中文平台（作为备选）
        elif "虎扑" in p or "hupu" in p:
            return f"https://s.hupu.com/all?q={kw_encoded}"
        elif "微博" in p or "weibo" in p:
            return f"https://s.weibo.com/weibo?q={kw_encoded}"
        elif "知乎" in p or "zhihu" in p:
            return f"https://www.zhihu.com/search?type=content&q={kw_encoded}"
        elif "b站" in p or "bilibili" in p:
            return f"https://search.bilibili.com/all?keyword={kw_encoded}"
        elif "抖音" in p or "douyin" in p:
            return f"https://www.douyin.com/search/{kw_encoded}"
        
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
        使用联网搜索获取真实的国外平台帖子/讨论链接
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
        
        search_prompt = f"""Based on the English forum post content and recommended topics, search and return 2 **real discussion/thread links** from ANY platform.

Post content (first 600 chars):
{text_content[:600]}

Recommended directions:
{chr(10).join(links_desc)}

Return JSON format:
[
  {{
    "title": "Actual discussion/thread title (in English)",
    "platform": "Platform name (can be ANY: Reddit/Twitter/YouTube/TikTok/虎扑/微博/知乎/B站/etc.)",
    "url": "Real thread/discussion URL (must be specific thread, not homepage or search page)",
    "search_keyword": "Precise search keyword (only if really can't find any specific thread link)"
  }}
]

**CRITICAL Requirements:**
1. **URL MUST be a SPECIFIC thread/discussion/video link, NOT homepage or search page**
2. **Priority order:**
   - First try: Reddit, Twitter, YouTube, TikTok (international platforms)
   - Second try: 虎扑, 微博, 知乎, B站 (Chinese platforms are OK if they have specific discussions)
   - Last resort: Other platforms are also acceptable
3. **Title should be in English** (translate if the actual discussion is in Chinese)
4. **Only use search_keyword if you absolutely cannot find ANY specific discussion link**
5. Prefer ANY specific thread URL over search pages"""

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
                        if any(indicator in url.lower() for indicator in ['/search?', '/search/', 'search_query=', 'search_result', '?q=']):
                            print(f"⚠️ Filtered search page: {url[:60]}...")
                            continue
                        
                        platform = result.get("platform", "Web")
                        valid_links.append({
                            "title": result.get("title", ""),
                            "platform": platform,
                            "url": url
                        })
                        # 区分国外和中文平台
                        platform_type = "🌍 International" if any(p in platform.lower() for p in ['reddit', 'twitter', 'youtube', 'tiktok', 'quora']) else "🇨🇳 Chinese"
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
        
        # 平台映射（论坛风格）
        platform_map = {
            "虎扑": "Reddit",
            "hupu": "Reddit",
            "微博": "Twitter",
            "weibo": "Twitter",
            "知乎": "Reddit",
            "b站": "YouTube",
            "bilibili": "YouTube",
            "抖音": "TikTok"
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
            if "keyword=" in original_url or "?q=" in original_url or "/search" in original_url:
                try:
                    import urllib.parse
                    parsed = urllib.parse.urlparse(original_url)
                    params = urllib.parse.parse_qs(parsed.query)
                    if 'keyword' in params:
                        search_keyword = params['keyword'][0]
                    elif 'q' in params:
                        search_keyword = params['q'][0]
                    elif 'wd' in params:
                        search_keyword = params['wd'][0]
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
        print("🔍 Attempting to retrieve real international discussion links...")
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
    
    def generate_english_html(self, english_text, image_paths, adapted_links, style_info, output_path):
        """生成英文版 HTML - 论坛风格"""
        
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
        
        # 获取风格标记
        style_name = style_info.get("name", "Discussion") if isinstance(style_info, dict) else "Discussion"
        style_badge = "🗣️" if "casual" in style_name.lower() else ("💭" if "analytical" in style_name.lower() else "🔥")
        
        # HTML 模板 - Reddit/论坛风格
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forum Discussion - English Version</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.8; max-width: 850px; margin: 0 auto; padding: 15px; background: #f8f9fa; color: #333; }}
        .post-container {{ background: white; border-radius: 10px; padding: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin: 10px 0; }}
        
        /* Header with style badge */
        .post-header {{ display: flex; align-items: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid #f0f0f0; }} 
        .avatar {{ width: 48px; height: 48px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin-right: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }} 
        .user-info {{ flex: 1; }}
        .user-info h3 {{ margin: 0; font-size: 18px; font-weight: 600; }} 
        .post-time {{ color: #999; font-size: 13px; margin-top: 4px; }} 
        .style-badge {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-left: 10px; display: inline-block; }}
        
        /* Content styling - forum style */
        .post-content {{ font-size: 16px; color: #2c3e50; letter-spacing: 0.02em; line-height: 1.9; }}
        .post-content p {{ margin: 1em 0; text-align: left; }}
        .post-content strong {{ color: #000; font-weight: 700; background: linear-gradient(to bottom, transparent 60%, #fff3cd 60%); }}
        .post-content h3 {{ font-size: 1.2em; margin-top: 1.5em; margin-bottom: 0.5em; color: #1a1a1a; }}
        
        .post-image {{ margin: 20px -25px; width: calc(100% + 50px); text-align: center; }}
        .post-image img {{ width: 100%; display: block; border-radius: 8px; }}
        .image-caption {{ color: #999; font-size: 13px; margin-top: 8px; font-style: italic; padding: 0 25px; }}

        /* Link card styling - forum style */
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
        .link-card.platform-reddit {{ border-left: 5px solid #FF4500; }}
        .link-card.platform-twitter {{ border-left: 5px solid #1DA1F2; }}
        .link-card.platform-youtube {{ border-left: 5px solid #FF0000; }}
        .link-card.platform-tiktok {{ border-left: 5px solid #000000; }}
        .link-card.platform-quora {{ border-left: 5px solid #B92B27; }}
        .link-card.platform-hupu {{ border-left: 5px solid #ff6600; }}
        .link-card.platform-weibo {{ border-left: 5px solid #ea5d5c; }}
        .link-card.platform-zhihu {{ border-left: 5px solid #0084ff; }}
        .link-card.platform-bilibili {{ border-left: 5px solid #23ade5; }}
        .link-card.platform-douyin {{ border-left: 5px solid #1c1e21; }}
        
        .link-info {{ flex: 1; }}
        .link-platform-tag {{ 
            font-size: 12px; font-weight: bold; margin-bottom: 4px; display: inline-block; padding: 2px 6px; border-radius: 4px; color: white;
        }}
        .tag-reddit {{ background: #FF4500; }}
        .tag-twitter {{ background: #1DA1F2; }}
        .tag-youtube {{ background: #FF0000; }}
        .tag-tiktok {{ background: #000; }}
        .tag-quora {{ background: #B92B27; }}
        .tag-hupu {{ background: #ff6600; }}
        .tag-weibo {{ background: #ea5d5c; }}
        .tag-zhihu {{ background: #0084ff; }}
        .tag-bilibili {{ background: #23ade5; }}
        .tag-douyin {{ background: #000; }}
        
        .link-title {{ font-weight: bold; color: #333; font-size: 15px; margin-top: 2px; }}
        .link-action {{ color: #999; font-size: 12px; margin-top: 4px; }}
        .link-icon {{ font-size: 24px; margin-right: 15px; }}

        .footer {{ margin-top:30px; border-top:1px solid #eee; padding-top:15px; color:#ccc; font-size:12px; text-align:center; }}

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
                <h3>Forum Member <span class="style-badge">{style_badge} {style_name.split('(')[0].strip()}</span></h3>
                <div class="post-time">{datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
            </div>
        </div>
        <div class="post-content">{html_content}</div>
        <div class="footer">
            Generated by AI • English Version • {len(image_paths)} Images • {len(adapted_links)} Links
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
        title = link_data.get('title', 'Related Discussion')
        platform = link_data.get('platform', 'Web').strip()
        url = link_data.get('url', '#')
        
        css_class = "platform-other"
        tag_class = "tag-other"
        icon = "🔗"
        
        p = platform.lower()
        # 国外平台
        if "reddit" in p:
            css_class = "platform-reddit"
            tag_class = "tag-reddit"
            icon = "💬"
        elif "twitter" in p or "x" == p:
            css_class = "platform-twitter"
            tag_class = "tag-twitter"
            icon = "🐦"
        elif "youtube" in p:
            css_class = "platform-youtube"
            tag_class = "tag-youtube"
            icon = "📺"
        elif "tiktok" in p:
            css_class = "platform-tiktok"
            tag_class = "tag-tiktok"
            icon = "🎵"
        elif "quora" in p:
            css_class = "platform-quora"
            tag_class = "tag-quora"
            icon = "❓"
        # 中文平台（作为备选）
        elif "虎扑" in p or "hupu" in p:
            css_class = "platform-hupu"
            tag_class = "tag-hupu"
            icon = "🏀"
        elif "微博" in p or "weibo" in p:
            css_class = "platform-weibo"
            tag_class = "tag-weibo"
            icon = "👁️"
        elif "知乎" in p or "zhihu" in p:
            css_class = "platform-zhihu"
            tag_class = "tag-zhihu"
            icon = "❓"
        elif "b站" in p or "bilibili" in p:
            css_class = "platform-bilibili"
            tag_class = "tag-bilibili"
            icon = "📺"
        elif "抖音" in p or "douyin" in p:
            css_class = "platform-douyin"
            tag_class = "tag-douyin"
            icon = "🎵"
        
        return f'''
<a href="{url}" class="link-card {css_class}" target="_blank">
    <div class="link-icon">{icon}</div>
    <div class="link-info">
        <span class="link-platform-tag {tag_class}">{platform}</span>
        <div class="link-title">{title}</div>
        <div class="link-action">Check out the discussion on {platform} &gt;</div>
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
        
        post_data = data.get("discussion_post", {})
        hot_topics = post_data.get("hot_topics", [])
        images = post_data.get("images", [])
        style = post_data.get("style", {})
        
        # 检查是否有reflection历史，如果有则使用最高分版本的HTML
        reflection_history = post_data.get("reflection_history")
        if reflection_history and len(reflection_history) > 0:
            print(f"🔍 检测到Reflection历史（{len(reflection_history)}个版本），选择最高分版本...")
            
            # 找到最高分版本
            best_version = max(reflection_history, key=lambda x: x.get("groupscore", 0))
            best_html_path = best_version["html_path"]
            best_score = best_version["groupscore"]
            best_version_name = best_version["version"]
            
            print(f"   ✅ 选择版本 {best_version_name} (GroupScore: {best_score:.4f})")
            
            # 从最高分HTML中提取内容
            chinese_text, links = self._extract_from_html(best_html_path)
            print(f"   📝 从HTML提取: {len(chinese_text)} 字符, {len(links)} 个链接")
        else:
            # 没有reflection，使用原始数据
            print(f"ℹ️  无Reflection历史，使用原始数据")
            chinese_text = post_data.get("text", "")
            links = post_data.get("links", [])
        
        if not chinese_text:
            print(f"❌ 没有文本内容")
            return None
        
        print(f"📝 原始文本长度: {len(chinese_text)} 字符")
        print(f"🔥 热点话题: {len(hot_topics)} 个")
        if links:
            print(f"🔗 原始链接: {[l.get('title', '') for l in links]}")
        
        # 转换为英文
        print("🔄 Converting to English (forum style)...")
        english_text, english_link_titles = self.convert_to_english(chinese_text, hot_topics, links)
        
        if not english_text:
            print(f"❌ 转换失败")
            return None
        
        print(f"✅ 英文文本长度: {len(english_text)} 字符")
        if english_link_titles:
            print(f"🔗 英文链接标题: {english_link_titles}")
        
        # 适配链接（使用英文标题）
        adapted_links = self.adapt_links_to_english(links, english_link_titles)
        
        # 生成英文版 HTML
        user_path = os.path.join(self.generated_dir, user_dir)
        output_html = os.path.join(user_path, "discussion_post_english.html")
        
        print("📄 Generating English HTML...")
        self.generate_english_html(english_text, images, adapted_links, style, output_html)
        
        # 保存英文数据
        english_data = {
            "text": english_text,
            "images": images,
            "links": adapted_links,
            "hot_topics": hot_topics,
            "style": style,
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
    
    parser = argparse.ArgumentParser(description='将虎扑讨论帖转换为英文版（Reddit/Twitter论坛风格）')
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
    
    converter = HupuEnglishConverter()
    
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

