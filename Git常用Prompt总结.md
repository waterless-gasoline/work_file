# Git 常用 Prompt 总结

## 初始化与推送

| 场景 | Prompt |
|------|--------|
| 初始化仓库 | `在当前目录初始化 git 仓库，添加 .gitignore 排除大文件和临时数据，然后提交当前代码` |
| 关联远程仓库 | `git remote add origin <仓库地址>` |
| 提交并推送 | `git add . && git commit -m "描述" && git push origin master` |

## 日常操作

| 场景 | Prompt |
|------|--------|
| 查看状态 | `帮我查看 git 状态` |
| 查看历史 | `帮我查看提交记录` |
| 查看差异 | `帮我查看有哪些文件被修改了` |
| 小步提交 | `帮我提交当前改动，commit message 写"feat: xxx"` |
| 添加特定文件 | `帮我添加 steps/ 和 utils/ 目录下的 .py 文件` |

## 回退操作

| 场景 | Prompt |
|------|--------|
| 回退到上一版本 | `我改炸了，帮我回退到上一个可用的版本` |
| 查看历史并回退 | `帮我查看提交历史，然后回退到指定版本` |
| 强制回退 | `git reset --hard <commit-id>` |

## GitHub CLI 相关

| 场景 | Prompt |
|------|--------|
| 创建仓库并推送 | `gh repo create <仓库名> --public --source=. --push` |
| 检查认证状态 | `gh auth status` |
| 认证登录 | `gh auth login` |

## .gitignore

| 场景 | Prompt |
|------|--------|
| 创建 .gitignore | `帮我创建 .gitignore，排除 *.zip, *.mp4, output/, __pycache__/ 等` |
| 修改 .gitignore | `帮我修改 .gitignore，添加/移除 xxx` |

---

## 简单记忆

- **改前先提交**：每次改代码前习惯性 `git commit`
- **小步多次**：每完成一个小功能就 commit，方便精准回退
- **描述清晰**：commit message 用 `feat:` `fix:` `chore:` 前缀