"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Play, Search, Signal } from "lucide-react";

import { completeLesson, getAssignedLessons } from "@/api/lessonsProvider";
import type { AssignedLessonsResponse } from "@/api/lessonsProvider";
import type { AssignedLesson } from "@/api/quizProvider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";

const getErrorMessage = (err: unknown, fallback: string) => {
  if (err && typeof err === "object" && "response" in err) {
    return (err as { response?: { data?: { detail?: string } } }).response?.data?.detail || fallback;
  }
  if (err instanceof Error && err.message.trim()) {
    return err.message;
  }
  return fallback;
};

const FEATURED_IMAGE =
  "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?q=80&w=1200&auto=format&fit=crop";

const CARD_IMAGES = [
  "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=1200&auto=format&fit=crop",
  "https://images.unsplash.com/photo-1555949963-aa79dcee981c?q=80&w=1200&auto=format&fit=crop",
  "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?q=80&w=1200&auto=format&fit=crop",
];

const levelLabel = (level: number) => {
  if (level === 1) return "Beginner";
  if (level === 2) return "Intermediate";
  return "Advanced";
};

const roleLabel = (role: string) => {
  if (!role) return "Learning path";
  return role.charAt(0).toUpperCase() + role.slice(1);
};

const progressValue = (lesson: AssignedLesson) => {
  if (typeof lesson.progress === "number") {
    return Math.max(0, Math.min(100, Math.round(lesson.progress)));
  }

  if (lesson.chunk_count <= 0) {
    if (lesson.status === "completed") return 100;
    if (lesson.status === "in_progress") return 50;
    return 0;
  }

  return Math.round((lesson.completed_chunks / lesson.chunk_count) * 100);
};

const lessonDescription = (lesson: AssignedLesson) => {
  return `Continue your ${lesson.role} path with this ${levelLabel(lesson.level).toLowerCase()} lesson.`;
};

const featuredBadge = (lesson: AssignedLesson) => {
  if (lesson.status === "completed") return "COMPLETED";
  if (lesson.status === "in_progress") return "IN PROGRESS";
  return "ASSIGNED";
};

export default function LessonsPage() {
  const router = useRouter();
  const [data, setData] = useState<AssignedLessonsResponse | null>(null);
  const [search, setSearch] = useState("");
  const [levelFilter, setLevelFilter] = useState<"All" | "Beginner" | "Intermediate" | "Advanced">("All");
  const [loading, setLoading] = useState(true);
  const [actionSource, setActionSource] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadLessons = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await getAssignedLessons();
        setData(response.data);
      } catch (err: unknown) {
        setError(getErrorMessage(err, "Failed to load lessons"));
      } finally {
        setLoading(false);
      }
    };

    loadLessons();
  }, []);

  const filteredLessons = useMemo(() => {
    const lessons = data?.lessons ?? [];
    const term = search.trim().toLowerCase();
    return lessons.filter((lesson) => {
      const matchesSearch =
        !term ||
        lesson.topic.toLowerCase().includes(term) ||
        lesson.role.toLowerCase().includes(term);
      const matchesLevel =
        levelFilter === "All" || levelLabel(lesson.level) === levelFilter;

      return matchesSearch && matchesLevel;
    });
  }, [data?.lessons, search, levelFilter]);

  const featuredLesson =
    filteredLessons.find((lesson) => lesson.status === "in_progress") ??
    filteredLessons.find((lesson) => lesson.status === "assigned") ??
    filteredLessons.find((lesson) => lesson.status !== "completed") ??
    filteredLessons[0] ??
    null;

  const previousLessons = filteredLessons.filter(
    (lesson) => lesson.status === "completed"
  );

  const upcomingLessons = (featuredLesson
    ? filteredLessons.filter((lesson) => lesson.source !== featuredLesson.source)
    : filteredLessons).filter((lesson) => lesson.status !== "completed");

  const syncLessonStatus = async (
    lesson: AssignedLesson,
    status: "assigned" | "in_progress" | "completed"
  ) => {
    setActionSource(lesson.source);
    setError(null);

    try {
      const response = await completeLesson({
        source: lesson.source,
        status,
      });
      setData(response.data);
      return true;
    } catch (err: unknown) {
      setError(getErrorMessage(err, "Failed to update lesson"));
      return false;
    } finally {
      setActionSource(null);
    }
  };

  const openLesson = async (lesson: AssignedLesson) => {
    const ok = await syncLessonStatus(lesson, "in_progress");
    if (!ok) return;
    router.push(`/lesson-details?source=${encodeURIComponent(lesson.source)}`);
  };

  const reviewLesson = (lesson: AssignedLesson) => {
    router.push(`/lesson-details?source=${encodeURIComponent(lesson.source)}`);
  };

  return (
    <main className="min-h-screen bg-background px-6 py-8 md:px-10">
      <div className="mx-auto max-w-7xl space-y-8">
        <div>
          <h1 className="text-4xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
            Explore Lessons
          </h1>
          <p className="mt-2 text-base text-slate-400 dark:text-slate-500">
            {data
              ? `Based on your ${data.role} path and current level ${levelLabel(data.level)}`
              : "Based on your completed lessons and level"}
          </p>
        </div>

        <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <Input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search for lessons, topics, or keywords..."
              className="h-11 rounded-xl border-slate-200 bg-white pl-10 shadow-sm dark:border-slate-800 dark:bg-black dark:text-slate-100"
            />
          </div>

          <div className="flex flex-wrap gap-3">
            <Button
              className="h-11 rounded-xl bg-blue-600 px-5 text-white hover:bg-blue-700"
              onClick={() => setSearch("")}
            >
              All Topics
            </Button>

            <Button
              variant="outline"
              className="h-11 rounded-xl border-slate-200 bg-white text-slate-500 dark:border-slate-800 dark:bg-black dark:text-slate-300"
              onClick={() =>
                setLevelFilter((current) =>
                  current === "All"
                    ? "Beginner"
                    : current === "Beginner"
                      ? "Intermediate"
                      : current === "Intermediate"
                        ? "Advanced"
                        : "All"
                )
              }
            >
              {levelFilter === "All" ? "Difficulty" : levelFilter}
            </Button>
          </div>
        </div>

        {loading ? (
          <Card className="rounded-2xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
            <CardContent className="p-6 text-slate-500 dark:text-slate-400">
              Fetching your lessons...
            </CardContent>
          </Card>
        ) : error ? (
          <Card className="rounded-2xl border-red-200 bg-white shadow-sm dark:border-red-900 dark:bg-black">
            <CardContent className="p-6">
              <p className="text-sm text-red-600">{error}</p>
            </CardContent>
          </Card>
        ) : filteredLessons.length === 0 ? (
          <Card className="rounded-2xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
            <CardContent className="p-6 text-slate-500 dark:text-slate-400">
              No lessons matched your search.
            </CardContent>
          </Card>
        ) : (
          <>
            {featuredLesson && (
              <Card className="rounded-2xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
                <CardContent className="p-6">
                  <div className="grid gap-6 lg:grid-cols-[1fr_320px] lg:items-center">
                    <div className="space-y-4">
                      <Badge className="rounded-full bg-blue-100 text-blue-700 hover:bg-blue-100">
                        {featuredBadge(featuredLesson)}
                      </Badge>

                      <p className="text-sm text-slate-400 dark:text-slate-500">
                        {roleLabel(data?.role ?? featuredLesson.role)} • {levelLabel(featuredLesson.level)}
                      </p>

                      <div>
                        <h2 className="text-4xl font-bold text-slate-900 dark:text-slate-100">
                          {featuredLesson.topic}
                        </h2>
                        <p className="mt-2 text-base text-slate-400 dark:text-slate-500">
                          {lessonDescription(featuredLesson)}
                        </p>
                      </div>

                      <div className="flex items-center gap-3">
                        <Progress
                          value={progressValue(featuredLesson)}
                          className="h-2.5 flex-1"
                        />
                        <span className="text-sm font-medium text-slate-600 dark:text-slate-300">
                          {progressValue(featuredLesson)}%
                        </span>
                      </div>

                      <Button
                        className="rounded-xl bg-blue-600 px-6 text-white hover:bg-blue-700"
                        onClick={() => openLesson(featuredLesson)}
                        disabled={actionSource === featuredLesson.source}
                      >
                        <Play className="mr-2 h-4 w-4" />
                        {actionSource === featuredLesson.source ? "Opening..." : "Resume lesson"}
                      </Button>
                    </div>

                    <div className="overflow-hidden rounded-2xl bg-slate-200">
                      <img
                        src={FEATURED_IMAGE}
                        alt="Featured lesson"
                        className="h-[240px] w-full object-cover"
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {upcomingLessons.length > 0 && (
              <section className="space-y-5">
                <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100">Upcoming Lessons</h2>

                <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
                  {upcomingLessons.map((lesson, index) => (
                    <Card
                      key={lesson.source}
                      className="overflow-hidden rounded-2xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black"
                    >
                      <div className="h-44 overflow-hidden bg-slate-200">
                        <img
                          src={CARD_IMAGES[index % CARD_IMAGES.length]}
                          alt={lesson.topic}
                          className="h-full w-full object-cover"
                        />
                      </div>

                      <CardContent className="space-y-3 p-4">
                        <div>
                          <h3 className="text-xl font-bold text-slate-900 dark:text-slate-100">
                            {lesson.topic}
                          </h3>
                          <p className="text-sm text-slate-400">{roleLabel(lesson.role)}</p>
                        </div>

                        <p className="text-sm leading-6 text-slate-500">
                          {lessonDescription(lesson)}
                        </p>

                        <div className="flex items-center pt-2 text-sm text-slate-400">
                          <div className="flex items-center gap-1">
                            <Signal className="h-4 w-4" />
                            <span>{levelLabel(lesson.level)}</span>
                          </div>
                        </div>

                        <Button
                          variant="outline"
                          className="w-full rounded-xl"
                          onClick={() => openLesson(lesson)}
                          disabled={actionSource === lesson.source}
                        >
                          {actionSource === lesson.source ? "Opening..." : "Open Lesson"}
                        </Button>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </section>
            )}

            {previousLessons.length > 0 && (
              <section className="space-y-5">
                <div className="flex items-end justify-between gap-3">
                  <div>
                    <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
                      Previous Lessons
                    </h2>
                    <p className="mt-1 text-sm text-slate-400 dark:text-slate-500">
                      Review lessons you already completed.
                    </p>
                  </div>
                  <Badge className="rounded-full bg-emerald-100 text-emerald-700 hover:bg-emerald-100">
                    {previousLessons.length} Completed
                  </Badge>
                </div>

                <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
                  {previousLessons.map((lesson, index) => (
                    <Card
                      key={lesson.source}
                      className="overflow-hidden rounded-2xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black"
                    >
                      <div className="h-44 overflow-hidden bg-slate-200">
                        <img
                          src={CARD_IMAGES[index % CARD_IMAGES.length]}
                          alt={lesson.topic}
                          className="h-full w-full object-cover"
                        />
                      </div>

                      <CardContent className="space-y-3 p-4">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <h3 className="text-xl font-bold text-slate-900 dark:text-slate-100">
                              {lesson.topic}
                            </h3>
                            <p className="text-sm text-slate-400">{roleLabel(lesson.role)}</p>
                          </div>
                          <Badge className="rounded-full bg-emerald-100 text-emerald-700 hover:bg-emerald-100">
                            Completed
                          </Badge>
                        </div>

                        <p className="text-sm leading-6 text-slate-500">
                          {lessonDescription(lesson)}
                        </p>

                        <div className="flex items-center justify-between pt-2 text-sm text-slate-400">
                          <div className="flex items-center gap-1">
                            <Signal className="h-4 w-4" />
                            <span>{levelLabel(lesson.level)}</span>
                          </div>
                          <span>100%</span>
                        </div>

                        <Progress value={100} className="h-2.5" />

                        <Button
                          variant="outline"
                          className="w-full rounded-xl"
                          onClick={() => reviewLesson(lesson)}
                        >
                          Review Lesson
                        </Button>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </section>
            )}
          </>
        )}
      </div>
    </main>
  );
}
