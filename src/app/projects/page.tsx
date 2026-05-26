import Link from "next/link";
import { projects } from "../data";

export default function ProjectsPage() {
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-col gap-12 px-6 py-10 sm:px-10 lg:px-12">
      <section className="rounded-[2rem] border border-white/10 bg-white/5 p-8 sm:p-12">
        <p className="text-sm uppercase tracking-[0.2em] text-blue-200">All Projects</p>
        <h1 className="mt-3 text-4xl font-semibold text-white">项目展示</h1>
        <p className="mt-4 max-w-3xl text-base leading-8 text-slate-300">
          这些项目覆盖电商 NLP、RAG、多模态生成与端侧视觉感知，重点展示业务问题、技术方案和量化结果。
        </p>
        <Link className="mt-6 inline-flex text-sm font-medium text-blue-200" href="/">
          ← 返回首页
        </Link>
      </section>

      <section className="grid gap-6">
        {projects.map((project) => (
          <article
            key={project.slug}
            className="rounded-3xl border border-white/10 bg-slate-900/70 p-7"
          >
            <div className="flex flex-col gap-6 lg:flex-row lg:justify-between">
              <div className="max-w-3xl">
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
                <h2 className="mt-5 text-2xl font-semibold text-white">{project.title}</h2>
                <p className="mt-3 text-sm leading-7 text-slate-300">{project.summary}</p>
                <p className="mt-4 text-sm leading-7 text-slate-400">
                  <span className="text-slate-200">场景：</span>
                  {project.scenario}
                </p>
                <div className="mt-5 flex flex-wrap gap-2 text-sm text-blue-100">
                  {project.techStack.map((tech) => (
                    <span
                      key={tech}
                      className="rounded-full border border-blue-400/20 bg-blue-400/10 px-3 py-1.5"
                    >
                      {tech}
                    </span>
                  ))}
                </div>
              </div>

              <div className="w-full max-w-xl rounded-3xl bg-white/5 p-5">
                <p className="text-sm font-medium text-white">关键指标</p>
                <ul className="mt-4 space-y-3 text-sm text-slate-300">
                  {project.metrics.map((metric) => (
                    <li key={metric} className="rounded-2xl border border-white/5 px-4 py-3">
                      {metric}
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="mt-6 flex items-center justify-between gap-4 border-t border-white/10 pt-6">
              <ul className="grid gap-2 text-sm text-slate-300">
                {project.highlights.slice(0, 2).map((highlight) => (
                  <li key={highlight}>• {highlight}</li>
                ))}
              </ul>
              <Link
                className="shrink-0 rounded-full border border-white/10 px-5 py-2.5 text-sm font-medium text-slate-100 transition hover:bg-white/5"
                href={`/projects/${project.slug}`}
              >
                查看详情
              </Link>
            </div>
          </article>
        ))}
      </section>
    </main>
  );
}
