import Link from "next/link";
import { notFound } from "next/navigation";
import { projects } from "../../data";

type ProjectPageProps = {
  params: Promise<{
    slug: string;
  }>;
};

export function generateStaticParams() {
  return projects.map((project) => ({
    slug: project.slug,
  }));
}

export default async function ProjectDetailPage({ params }: ProjectPageProps) {
  const { slug } = await params;
  const project = projects.find((item) => item.slug === slug);

  if (!project) {
    notFound();
  }

  return (
    <main className="mx-auto flex w-full max-w-5xl flex-col gap-10 px-6 py-10 sm:px-10 lg:px-12">
      <section className="rounded-[2rem] border border-white/10 bg-white/5 p-8 sm:p-12">
        <Link className="text-sm font-medium text-blue-200" href="/projects">
          ← 返回项目列表
        </Link>
        <div className="mt-6 flex flex-wrap gap-2">
          {project.tags.map((tag) => (
            <span
              key={tag}
              className="rounded-full bg-white/5 px-3 py-1 text-xs text-slate-300"
            >
              {tag}
            </span>
          ))}
        </div>
        <h1 className="mt-5 text-4xl font-semibold text-white">{project.title}</h1>
        <p className="mt-4 text-base leading-8 text-slate-300">{project.summary}</p>
        <p className="mt-4 text-sm leading-7 text-slate-400">
          <span className="text-slate-200">应用场景：</span>
          {project.scenario}
        </p>
      </section>

      <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <article className="rounded-3xl border border-white/10 bg-slate-900/70 p-7">
          <p className="text-sm uppercase tracking-[0.2em] text-blue-200">Background</p>
          <h2 className="mt-3 text-2xl font-semibold text-white">项目背景</h2>
          <p className="mt-4 text-sm leading-8 text-slate-300">{project.details.background}</p>
        </article>

        <article className="rounded-3xl border border-white/10 bg-slate-900/70 p-7">
          <p className="text-sm uppercase tracking-[0.2em] text-blue-200">Tech Stack</p>
          <h2 className="mt-3 text-2xl font-semibold text-white">技术栈</h2>
          <div className="mt-5 flex flex-wrap gap-2">
            {project.techStack.map((tech) => (
              <span
                key={tech}
                className="rounded-full border border-blue-400/20 bg-blue-400/10 px-3 py-1.5 text-sm text-blue-100"
              >
                {tech}
              </span>
            ))}
          </div>
        </article>
      </section>

      <section className="grid gap-6 lg:grid-cols-2">
        <article className="rounded-3xl border border-white/10 bg-white/5 p-7">
          <p className="text-sm uppercase tracking-[0.2em] text-blue-200">Solution</p>
          <h2 className="mt-3 text-2xl font-semibold text-white">方案设计</h2>
          <ul className="mt-5 space-y-4 text-sm leading-7 text-slate-300">
            {project.details.solution.map((item) => (
              <li key={item} className="rounded-2xl bg-slate-900/80 px-4 py-4">
                {item}
              </li>
            ))}
          </ul>
        </article>

        <article className="rounded-3xl border border-white/10 bg-white/5 p-7">
          <p className="text-sm uppercase tracking-[0.2em] text-blue-200">Results</p>
          <h2 className="mt-3 text-2xl font-semibold text-white">结果指标</h2>
          <ul className="mt-5 space-y-4 text-sm leading-7 text-slate-300">
            {project.details.results.map((item) => (
              <li key={item} className="rounded-2xl bg-slate-900/80 px-4 py-4">
                {item}
              </li>
            ))}
          </ul>
        </article>
      </section>

      <section className="rounded-3xl border border-white/10 bg-slate-900/70 p-7">
        <p className="text-sm uppercase tracking-[0.2em] text-blue-200">Highlights</p>
        <h2 className="mt-3 text-2xl font-semibold text-white">核心工作拆解</h2>
        <ul className="mt-5 grid gap-4 text-sm leading-7 text-slate-300 lg:grid-cols-3">
          {project.highlights.map((item) => (
            <li key={item} className="rounded-2xl bg-white/5 px-4 py-4">
              {item}
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
