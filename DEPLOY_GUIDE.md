# 🚀 GitHub Pages 部署指南

## 快速部署步骤

### 第一步：创建 GitHub 仓库

1. 登录 [github.com](https://github.com)
2. 点击右上角 `+` → `New repository`
3. 填写信息：
   - Repository name: `NightUAV-Sim`
   - Description: `A Synthetic Benchmark for Nighttime UAV 3D Reconstruction`
   - 选择 **Public**
   - ✅ Add a README file（取消勾选，我们已有）
4. 点击 `Create repository`

### 第二步：上传文件

**方式 A：网页上传（简单）**

1. 在仓库页面点击 `Add file` → `Upload files`
2. 解压下载的 `NightUAV-Sim.zip`
3. 将文件夹内的所有文件拖入上传区域：
   ```
   index.html
   README.md
   LICENSE
   .gitignore
   images/（文件夹）
   ```
4. 填写 Commit message: `Initial commit`
5. 点击 `Commit changes`

**方式 B：命令行上传（推荐）**

```bash
# 1. 克隆空仓库
git clone https://github.com/rebecca0011/NightUAV-Sim.git
cd NightUAV-Sim

# 2. 复制文件到仓库目录
# 将解压后的所有文件复制到这个目录

# 3. 提交并推送
git add .
git commit -m "Initial commit: project page"
git push origin main
```

### 第三步：启用 GitHub Pages

1. 进入仓库页面
2. 点击 `Settings`（⚙️ 齿轮图标）
3. 左侧菜单滚动到 `Pages`
4. 配置：
   - Source: `Deploy from a branch`
   - Branch: `main`
   - Folder: `/ (root)`
5. 点击 `Save`

### 第四步：等待部署

- 等待 1-2 分钟
- 刷新 Settings → Pages 页面
- 会显示：`Your site is live at https://rebecca0011.github.io/NightUAV-Sim/`

---

## 📝 部署后的修改

### 修改内容

1. 在 GitHub 仓库页面直接点击文件
2. 点击 ✏️ 编辑按钮
3. 修改后点击 `Commit changes`
4. 等待 1-2 分钟自动更新

### 添加图片

1. 准备好图片文件（建议 800×600 像素）
2. 进入 `images` 文件夹
3. 点击 `Add file` → `Upload files`
4. 上传图片
5. 修改 `index.html` 中的图片引用

---

## ⚠️ 需要修改的占位内容

在上传前，请修改以下内容：

### index.html 中

| 查找 | 替换为 |
|------|--------|
| `yourusername` | 你的 GitHub 用户名 |
| `your-email@example.com` | 你的邮箱地址 |
| `Author Name` | 你的真实姓名 |
| `~XX GB` | 实际数据大小 |

### README.md 中

| 查找 | 替换为 |
|------|--------|
| `yourusername` | 你的 GitHub 用户名 |
| `your-email@example.com` | 你的邮箱地址 |

---

## 🖼️ 添加图片示例

修改 `index.html` 中的图片占位符：

**修改前：**
```html
<div class="illumination-img">
    <span>[Noon Image]</span>
</div>
```

**修改后：**
```html
<div class="illumination-img">
    <img src="images/noon.jpg" alt="Noon lighting" style="width:100%; height:100%; object-fit:cover;">
</div>
```

---

## 🔗 最终网址

部署成功后，你的项目页面地址为：

```
https://rebecca0011.github.io/NightUAV-Sim/
```

例如：`https://zhangsan.github.io/NightUAV-Sim/`

---

## ❓ 常见问题

**Q: 页面显示 404？**
A: 确保 `index.html` 在仓库根目录，而不是在子文件夹中。

**Q: 修改后没有更新？**
A: GitHub Pages 有缓存，等待 2-5 分钟或清除浏览器缓存。

**Q: 图片不显示？**
A: 检查图片路径是否正确，注意大小写敏感。

---

祝你部署顺利！🎉
