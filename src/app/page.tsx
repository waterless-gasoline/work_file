import Link from "next/link";
import { focusAreas, metrics, profile, projects } from "./data";

const featuredProjects = projects.slice(0, 3);

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-col gap-24 px-6 py-10 sm:px-10 lg:px-12">
      <section className="rounded-[2rem] border border-white/10 bg-white/5 p-8 shadow-2xl shadow-blue-950/20 backdrop-blur sm:p-12">
        <div className="flex flex-col gap-10 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-3xl space-y-6">
            <span className="inline-flex rounded-full border border-blue-400/30 bg-blue-400/10 px-4 py-1 text-sm text-blue-200">
              AI 算法 / NLP / 多模态 / 大模型应用
            </span>
            <div className="space-y-4">
              <h1 className="text-4xl font-semibold tracking-tight text-white sm:text-5xl lg:text-6xl">
                {profile.name}
              </h1>
              <p className="text-lg text-blue-100/90 sm:text-xl">{profile.role}</p>
              <p className="max-w-2xl text-base leading-8 text-slate-300 sm:text-lg">
                {profile.intro}
              </p>
            </div>
            <div className="flex flex-wrap gap-3 text-sm text-slate-200">
              {profile.availability.map((item) => (
                <span
                  key={item}
                  className="rounded-full border border-white/10 bg-slate-900/70 px-4 py-2"
                >
                  {item}
                </span>
              ))}
            </div>
          </div>

          <div className="flex flex-col gap-3 sm:flex-row lg:flex-col">
            <a
              className="rounded-full bg-blue-500 px-6 py-3 text-center font-medium text-white transition hover:bg-blue-400"
              href="#projects"
            >
              查看项目
            </a>
            <a
              className="rounded-full border border-white/15 px-6 py-3 text-center font-medium text-slate-100 transition hover:bg-white/5"
              href={profile.contact.resume}
              target="_blank"
              rel="noopener noreferrer"
            >
              下载简历
            </a>
            <a
              className="rounded-full border border-white/15 px-6 py-3 text-center font-medium text-slate-100 transition hover:bg-white/5"
              href={profile.contact.github}
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </a>
          </div>
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {metrics.map((metric) => (
          <article
            key={metric.label}
            className="rounded-3xl border border-white/10 bg-slate-900/70 p-6"
          >
            <p className="text-sm text-slate-400">{metric.label}</p>
            <p className="mt-3 text-3xl font-semibold text-white">{metric.value}</p>
            <p className="mt-2 text-sm leading-6 text-slate-300">{metric.detail}</p>
          </article>
        ))}
      </section>

      <section className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-3xl border border-white/10 bg-white/5 p-8">
          <p className="text-sm uppercase tracking-[0.2em] text-blue-200">Focus</p>
          <h2 className="mt-3 text-2xl font-semibold text-white">重点能力方向</h2>
          <div className="mt-6 flex flex-wrap gap-3">
            {focusAreas.map((area) => (
              <span
                key={area}
                className="rounded-full border border-blue-400/20 bg-blue-400/10 px-4 py-2 text-sm text-blue-100"
              >
                {area}
              </span>
            ))}
          </div>
        </div>

        <div className="rounded-3xl border border-white/10 bg-slate-900/80 p-8">
          <p className="text-sm uppercase tracking-[0.2em] text-blue-200">Contact</p>
          <h2 className="mt-3 text-2xl font-semibold text-white">联系我</h2>
          <div className="mt-6 space-y-4 text-sm text-slate-300">
            <p>
              邮箱：<span className="text-white">{profile.contact.email}</span>
            </p>
            <p>
              电话：<span className="text-white">{profile.contact.phone}</span>
            </p>
            <p className="leading-7">
              目前先上线第一版作品集框架，后续会继续补充项目代码、效果截图和更细的技术拆解。
            </p>
          </div>
        </div>
      </section>

      <section id="projects" className="space-y-8">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-blue-200">Projects</p>
            <h2 className="mt-3 text-3xl font-semibold text-white">精选项目</h2>
            <p className="mt-3 max-w-2xl text-slate-300">
              围绕电商 NLP、大模型应用、多模态生成与端侧部署展开，强调业务场景、技术方案与结果指标。
            </p>
          </div>
          <Link
            className="text-sm font-medium text-blue-200 transition hover:text-blue-100"
            href="/projects"
          >
            查看全部项目 →
          </Link>
        </div>

        <div className="grid gap-6 xl:grid-cols-3">
          {featuredProjects.map((project) => (
            <article
              key={project.slug}
              className="rounded-3xl border border-white/10 bg-slate-900/70 p-6"
            >
              <div className="flex flex-wrap gap-2">
                {project.tags.map((tag) => (
                  <span
                    key={tag}
                    className="rounded-full bg-white/5 px-3 py-1 text-xs text-slate-300"
                  >
                    {tag}
                  </span>
                ))}
              </div>
              <h3 className="mt-5 text-2xl font-semibold text-white">{project.title}</h3>
              <p className="mt-3 text-sm leading-7 text-slate-300">{project.summary}</p>
              <ul className="mt-5 space-y-3 text-sm text-slate-300">
                {project.metrics.slice(0, 3).map((metric) => (
                  <li key={metric} className="rounded-2xl bg-white/5 px-4 py-3">
                    {metric}
                  </li>
                ))}
              </ul>
              <Link
                className="mt-6 inline-flex text-sm font-medium text-blue-200 transition hover:text-blue-100"
                href={`/projects/${project.slug}`}
              >
                查看详情 →
              </Link>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
