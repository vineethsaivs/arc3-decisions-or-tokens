import html, http.cookiejar, json, re, sys, urllib.request
cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
op.addheaders = [("User-Agent", "Mozilla/5.0")]
op.open("https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion", timeout=60).read()
tok = next(c.value for c in cj if c.name == "XSRF-TOKEN")
def rpc(m, b):
    r = urllib.request.Request("https://www.kaggle.com/api/i/discussions.DiscussionsService/" + m, data=json.dumps(b).encode(), headers={"X-XSRF-TOKEN": tok, "Content-Type": "application/json"})
    return json.load(op.open(r, timeout=60))
t = lambda s: html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()
for tid in map(int, sys.argv[1:]):
    d = rpc("GetForumTopicById", {"forumTopicId": tid, "includeComments": True})["forumTopic"]
    print("##", tid, d.get("name") or d.get("title"))
    fm = d.get("firstMessage") or {}
    print(fm.get("postDate","")[:16], (fm.get("author") or {}).get("displayName"), ":", (fm.get("rawMarkdown") or t(fm.get("content")))[:800].replace("\u2014"," - ").replace("\u2013","-"))
    def walk(cs, depth=0):
        for c in cs:
            print("  "*depth+"-", c.get("postDate","")[:16], (c.get("author") or {}).get("displayName"), ":", (c.get("rawMarkdown") or t(c.get("content")))[:700].replace("\u2014"," - ").replace("\u2013","-").replace("\n"," "))
            walk(c.get("replies") or [], depth+1)
    walk(d.get("comments", []))
