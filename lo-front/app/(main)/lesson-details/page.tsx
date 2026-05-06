"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Compass,
  Lightbulb,
  MessageSquareQuote,
  PartyPopper,
  PencilLine,
  Signal,
  Sparkles,
} from "lucide-react";

import {
  completeLesson,
  getLessonDetail,
  getLessonState,
  saveLessonState,
} from "@/api/lessonsProvider";
import type {
  AssignedLessonsResponse,
  LessonDetailResponse,
  LessonQuizQuestion,
  LessonNextRecommendation,
  LessonStoryboardSlide,
} from "@/api/lessonsProvider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
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

const levelLabel = (level: number) => {
  if (level === 1) return "Beginner";
  if (level === 2) return "Intermediate";
  return "Advanced";
};

const LESSON_PASSING_PERCENT = 70;

const statusLabel = (status: string) => {
  if (!status) return "Assigned";
  return status.replace("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
};

const illustrationPalette = (slideIndex: number) => {
  const palettes = [
    "from-amber-100 via-orange-50 to-white",
    "from-blue-100 via-cyan-50 to-white",
    "from-emerald-100 via-lime-50 to-white",
    "from-rose-100 via-pink-50 to-white",
  ];
  return palettes[slideIndex % palettes.length];
};

const buildNextLessonFromAssignments = (
  lessonSource: string,
  payload: AssignedLessonsResponse
): LessonNextRecommendation | null => {
  const nextLesson = payload.lessons.find(
    (item) => item.source !== lessonSource && item.status !== "completed"
  );

  if (!nextLesson) {
    return null;
  }

  return {
    source: nextLesson.source,
    title: nextLesson.topic,
    reason: "You passed this lesson, so this is the best next step in your current path.",
    level: nextLesson.level,
  };
};

export default function LessonPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const source = searchParams.get("source") || "";

  const [lesson, setLesson] = useState<LessonDetailResponse | null>(null);
  const [activeSlide, setActiveSlide] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [completing, setCompleting] = useState(false);
  const [quizAnswers, setQuizAnswers] = useState<Record<string, number>>({});
  const [quizSubmitted, setQuizSubmitted] = useState(false);
  const [phase, setPhase] = useState<"story" | "quiz" | "next">("story");

  useEffect(() => {
    const loadLesson = async () => {
      if (!source) {
        setError("Missing lesson source");
        setLoading(false);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const [detailResponse, stateResponse] = await Promise.all([
          getLessonDetail(source),
          getLessonState(source),
        ]);
        const lessonData = detailResponse.data;
        const stateData = stateResponse.data;

        setLesson(lessonData);
        setActiveSlide(
          Math.max(0, Math.min(stateData.active_slide || 0, lessonData.slides.length - 1))
        );
        setQuizAnswers(stateData.quiz_answers || {});
        setQuizSubmitted(Boolean(stateData.quiz_submitted));
        setPhase(
          lessonData.status === "completed" && stateData.phase === "next"
            ? "next"
            : stateData.phase === "quiz"
              ? "quiz"
              : "story"
        );
      } catch (err: unknown) {
        setError(getErrorMessage(err, "Failed to load lesson"));
      } finally {
        setLoading(false);
      }
    };

    loadLesson();
  }, [source]);

  useEffect(() => {
    if (!lesson || !source || loading) return;

    const timeout = window.setTimeout(() => {
      void saveLessonState({
        source,
        active_slide: activeSlide,
        phase,
        quiz_answers: quizAnswers,
        quiz_submitted: quizSubmitted,
      });
    }, 250);

    return () => window.clearTimeout(timeout);
  }, [lesson, source, loading, activeSlide, phase, quizAnswers, quizSubmitted]);

  const slide = lesson?.slides[activeSlide] || null;
  const slideProgress = useMemo(() => {
    if (!lesson?.slides?.length) return 0;
    return Math.round(((activeSlide + 1) / lesson.slides.length) * 100);
  }, [activeSlide, lesson]);

  const quizScore = useMemo(() => {
    if (!lesson?.end_quiz?.length) return 0;
    return lesson.end_quiz.reduce((total, question, index) => {
      return total + (quizAnswers[index] === question.correct_index ? 1 : 0);
    }, 0);
  }, [lesson, quizAnswers]);

  const passingScore = useMemo(() => {
    if (!lesson?.end_quiz?.length) return 0;
    return Math.ceil((lesson.end_quiz.length * LESSON_PASSING_PERCENT) / 100);
  }, [lesson]);

  const quizPassed = lesson ? quizScore >= passingScore : false;

  const submitQuiz = async () => {
    if (!lesson) return;
    setQuizSubmitted(true);

    if (Object.keys(quizAnswers).length < lesson.end_quiz.length) {
      setError("Please answer every quiz question before finishing the lesson.");
      return;
    }

    setError(null);
    if (!quizPassed) {
      setCompleting(false);
      return;
    }

    setCompleting(true);
    try {
      const completionResponse = await completeLesson({
        source: lesson.source,
        status: "completed",
        quiz_score: quizScore,
        total_questions: lesson.end_quiz.length,
      });
      const nextLesson = buildNextLessonFromAssignments(
        lesson.source,
        completionResponse.data
      );
      setLesson((current) =>
        current
          ? {
              ...current,
              status: "completed",
              progress: 100,
              next_lesson: nextLesson,
            }
          : current
      );
      setQuizSubmitted(true);
      setPhase("next");
    } catch (err: unknown) {
      setError(getErrorMessage(err, "Failed to complete lesson"));
    } finally {
      setCompleting(false);
    }
  };

  const retryQuiz = () => {
    setQuizSubmitted(false);
    setQuizAnswers({});
    setError(null);
  };

  if (loading) {
    return (
      <main className="min-h-screen bg-background px-4 py-8 md:px-8">
        <div className="mx-auto max-w-7xl">
          <Card className="rounded-2xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
            <CardContent className="p-6 text-slate-500 dark:text-slate-400">
              Building your personalized lesson...
            </CardContent>
          </Card>
        </div>
      </main>
    );
  }

  if (error || !lesson || !slide) {
    return (
      <main className="min-h-screen bg-background px-4 py-8 md:px-8">
        <div className="mx-auto max-w-7xl">
          <Card className="rounded-2xl border-red-200 bg-white shadow-sm dark:border-red-900 dark:bg-black">
            <CardContent className="p-6">
              <p className="text-sm text-red-600">{error || "Failed to load lesson"}</p>
            </CardContent>
          </Card>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-background px-4 py-8 md:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <nav className="flex items-center gap-2 text-xs text-slate-400">
          <button
            className="transition-colors hover:text-slate-600"
            onClick={() => router.push("/lessons")}
          >
            Lessons
          </button>
          <ChevronRight size={10} />
          <span>{lesson.role.charAt(0).toUpperCase() + lesson.role.slice(1)} Path</span>
          <ChevronRight size={10} />
          <span className="font-medium text-slate-600">{lesson.topic}</span>
        </nav>

        <div className="space-y-6">
            <Card className="overflow-hidden rounded-3xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
              <CardContent className="p-0">
                <div className="border-b border-slate-100 px-6 py-5 md:px-8 dark:border-slate-800">
                  <div className="flex flex-wrap items-center gap-3 text-[11px] font-semibold uppercase tracking-[0.18em]">
                    {phase === "story" && (
                      <Badge className="rounded-full bg-blue-100 text-blue-700 hover:bg-blue-100">
                        Slide {activeSlide + 1}
                      </Badge>
                    )}
                    {phase === "quiz" && (
                      <Badge className="rounded-full bg-amber-100 text-amber-700 hover:bg-amber-100">
                        End Quiz
                      </Badge>
                    )}
                    {phase === "next" && (
                      <Badge className="rounded-full bg-emerald-100 text-emerald-700 hover:bg-emerald-100">
                        Next Step
                      </Badge>
                    )}
                    <span className="flex items-center gap-1 text-slate-400 dark:text-slate-500">
                      <Clock3 className="h-3.5 w-3.5" />
                      {lesson.estimated_minutes} min
                    </span>
                    <span className="flex items-center gap-1 text-slate-400 dark:text-slate-500">
                      <Signal className="h-3.5 w-3.5" />
                      {levelLabel(lesson.level)}
                    </span>
                    <span className="flex items-center gap-1 text-slate-400 dark:text-slate-500">
                      <Sparkles className="h-3.5 w-3.5" />
                      {statusLabel(lesson.status)}
                    </span>
                  </div>

                  <h1 className="mt-4 text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100 md:text-4xl">
                    {lesson.topic}
                  </h1>
                  {phase === "story" && (
                    <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-500 dark:text-slate-400 md:text-base">
                      {lesson.overview}
                    </p>
                  )}
                  {phase === "quiz" && (
                    <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-500 dark:text-slate-400 md:text-base">
                      You reached the end of the storybook. Let’s do a short quiz to confirm the key ideas before unlocking the next lesson.
                    </p>
                  )}
                  {phase === "next" && (
                    <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-500 dark:text-slate-400 md:text-base">
                      Nice work. Your quiz is complete, and the next lesson is now ready from inside this learning flow.
                    </p>
                  )}
                </div>

                {phase === "story" && (
                  <div className="grid gap-6 px-6 py-6 md:px-8 lg:grid-cols-[1.15fr_0.85fr]">
                    <div className="space-y-5">
                      <div
                        className={`relative overflow-hidden rounded-3xl border border-slate-200 bg-gradient-to-br dark:border-slate-800 dark:from-slate-950 dark:via-slate-950 dark:to-black ${illustrationPalette(
                          activeSlide
                        )} p-6`}
                      >
                        <div className="absolute -right-10 -top-10 h-32 w-32 rounded-full bg-white/70 blur-2xl dark:bg-blue-950/30" />

                        <div className="relative flex min-h-[360px] flex-col justify-between">
                          <div className="space-y-5">
                            {slide.scene_caption && (
                              <p className="max-w-xl text-sm font-medium italic text-slate-500 dark:text-slate-400">
                                {slide.scene_caption}
                              </p>
                            )}
                            <h2 className="max-w-xl text-3xl font-bold leading-tight text-slate-900 dark:text-slate-100">
                              {slide.title}
                            </h2>
                            <p className="max-w-2xl text-base leading-8 text-slate-700 dark:text-slate-300">
                              {slide.narrative}
                            </p>
                          </div>

                          {slide.dialogue_line && (
                            <div className="max-w-lg rounded-[2rem] rounded-bl-md border border-white/70 bg-white/90 px-5 py-4 text-sm leading-6 text-slate-700 shadow-sm dark:border-slate-700 dark:bg-slate-950/95 dark:text-slate-200">
                              {slide.dialogue_line}
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="grid gap-4 lg:grid-cols-2">
                        <Card className="rounded-2xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
                          <CardContent className="p-5">
                            <div className="flex items-center gap-2">
                              <BookOpen className="h-4 w-4 text-blue-600" />
                              <h3 className="text-base font-bold text-slate-900 dark:text-slate-100">
                                Key Takeaways
                              </h3>
                            </div>
                            <div className="mt-4 space-y-3">
                              {slide.bullets.map((bullet, index) => (
                                <div key={`${bullet}-${index}`} className="rounded-2xl bg-slate-50 px-4 py-3 dark:bg-slate-950">
                                  <p className="text-sm leading-6 text-slate-700 dark:text-slate-300">{bullet}</p>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>

                        <Card className="rounded-2xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
                          <CardContent className="p-5">
                            <div className="flex items-center gap-2">
                              <Lightbulb className="h-4 w-4 text-amber-500" />
                              <h3 className="text-base font-bold text-slate-900 dark:text-slate-100">
                                Lesson Objectives
                              </h3>
                            </div>
                            <div className="mt-4 space-y-3">
                              {lesson.learning_objectives.map((objective, index) => (
                                <div key={`${objective}-${index}`} className="rounded-2xl border border-slate-200 px-4 py-3 dark:border-slate-800 dark:bg-slate-950">
                                  <p className="text-sm leading-6 text-slate-700 dark:text-slate-300">{objective}</p>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      </div>

                      {(slide.speaker_note || slide.checkpoint_question) && (
                        <Card className="rounded-2xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
                          <CardContent className="grid gap-4 p-5 md:grid-cols-2">
                            {slide.speaker_note && (
                              <div className="space-y-3">
                                <div className="flex items-center gap-2">
                                  <MessageSquareQuote className="h-4 w-4 text-violet-500" />
                                  <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                                    Teaching Note
                                  </h3>
                                </div>
                                <p className="text-sm leading-6 text-slate-600 dark:text-slate-300">
                                  {slide.speaker_note}
                                </p>
                              </div>
                            )}

                            {slide.checkpoint_question && (
                              <div className="rounded-2xl bg-slate-50 p-4 dark:bg-slate-950">
                                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400 dark:text-slate-500">
                                  Pause And Reflect
                                </p>
                                <p className="mt-2 text-sm leading-6 text-slate-700 dark:text-slate-300">
                                  {slide.checkpoint_question}
                                </p>
                              </div>
                            )}
                          </CardContent>
                        </Card>
                      )}

                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div className="flex flex-wrap items-center gap-3">
                          <Button
                            variant="outline"
                            className="rounded-xl border-slate-200 bg-white"
                            onClick={() => setActiveSlide((current) => Math.max(0, current - 1))}
                            disabled={activeSlide === 0}
                          >
                            <ArrowLeft className="mr-2 h-4 w-4" />
                            Previous
                          </Button>

                          <Button
                            variant="outline"
                            className="rounded-xl border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100 dark:border-blue-900 dark:bg-blue-950/30 dark:text-blue-300 dark:hover:bg-blue-950/50"
                            onClick={() =>
                              router.push(`/chatbot?source=${encodeURIComponent(lesson.source)}`)
                            }
                          >
                            Ask AI About This Lesson
                          </Button>
                        </div>

                        {activeSlide === lesson.slides.length - 1 ? (
                          <Button
                            className="rounded-xl bg-blue-600 text-white hover:bg-blue-700"
                            onClick={() => setPhase("quiz")}
                          >
                            <PencilLine className="mr-2 h-4 w-4" />
                            Continue To Quiz
                          </Button>
                        ) : (
                          <Button
                            className="rounded-xl bg-blue-600 text-white hover:bg-blue-700"
                            onClick={() =>
                              setActiveSlide((current) =>
                                Math.min(lesson.slides.length - 1, current + 1)
                              )
                            }
                          >
                            Next Slide
                            <ArrowRight className="ml-2 h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    </div>

                    <div className="space-y-5">
                        <Card className="rounded-2xl border-slate-200 bg-slate-50 shadow-none dark:border-slate-800 dark:bg-slate-950">
                          <CardContent className="p-5">
                            <div className="flex items-center justify-between gap-3">
                            <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">Slide Progress</h3>
                            <span className="text-xs font-semibold text-blue-600">
                              {slideProgress}%
                            </span>
                          </div>
                          <Progress value={slideProgress} className="mt-4 h-2.5" />
                          <p className="mt-3 text-xs text-slate-400 dark:text-slate-500">
                            Slide {activeSlide + 1} of {lesson.slides.length}
                          </p>
                        </CardContent>
                      </Card>

                      <Card className="rounded-2xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
                        <CardContent className="p-5">
                          <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">Storybook</h3>
                          <div className="mt-4 divide-y divide-slate-100 dark:divide-slate-800">
                            {lesson.slides.map((item, index) => (
                              <SlideListItem
                                key={`${item.title}-${index}`}
                                slide={item}
                                index={index}
                                active={index === activeSlide}
                                onClick={() => setActiveSlide(index)}
                              />
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  </div>
                )}

                {phase === "quiz" && (
                  <div className="px-6 py-6 md:px-8">
                    <div className="mx-auto max-w-3xl space-y-6">
                      <div className="flex items-center justify-between gap-3">
                        <Button
                          variant="outline"
                          className="rounded-xl border-slate-200 bg-white dark:border-slate-800 dark:bg-black dark:text-slate-100"
                          onClick={() => setPhase("story")}
                        >
                          <ArrowLeft className="mr-2 h-4 w-4" />
                          Back To Lesson
                        </Button>
                        <span className="text-sm text-slate-400 dark:text-slate-500">
                          {lesson.end_quiz.length} questions
                        </span>
                      </div>

                      {lesson.end_quiz.map((question, questionIndex) => (
                        <QuizCard
                          key={`${question.question}-${questionIndex}`}
                          question={question}
                          index={questionIndex}
                          selectedIndex={quizAnswers[questionIndex]}
                          showResult={quizSubmitted}
                          onSelect={(answerIndex) =>
                            setQuizAnswers((current) => ({
                              ...current,
                              [questionIndex]: answerIndex,
                            }))
                          }
                        />
                      ))}

                      <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl bg-slate-50 p-4 dark:bg-slate-950">
                        <div>
                          <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                            Ready to finish this lesson?
                          </p>
                          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                            Score at least {passingScore}/{lesson.end_quiz.length} to unlock the next lesson.
                          </p>
                        </div>

                        <Button
                          className="rounded-xl bg-blue-600 text-white hover:bg-blue-700"
                          onClick={submitQuiz}
                          disabled={completing}
                        >
                          <CheckCircle2 className="mr-2 h-4 w-4" />
                          {completing ? "Saving..." : "Submit Quiz"}
                        </Button>
                      </div>

                      {quizSubmitted && (
                        <div
                          className={
                            quizPassed
                              ? "rounded-2xl border border-emerald-200 bg-emerald-50 p-5 dark:border-emerald-900 dark:bg-emerald-950/30"
                              : "rounded-2xl border border-amber-200 bg-amber-50 p-5 dark:border-amber-900 dark:bg-amber-950/30"
                          }
                        >
                          <div className="flex items-center gap-3">
                            <PartyPopper
                              className={quizPassed ? "h-5 w-5 text-emerald-600" : "h-5 w-5 text-amber-600"}
                            />
                            <div>
                              <p
                                className={
                                  quizPassed
                                    ? "text-sm font-semibold text-emerald-900 dark:text-emerald-300"
                                    : "text-sm font-semibold text-amber-900 dark:text-amber-300"
                                }
                              >
                                Quiz score: {quizScore}/{lesson.end_quiz.length}
                              </p>
                              <p
                                className={
                                  quizPassed
                                    ? "mt-1 text-sm text-emerald-700 dark:text-emerald-400"
                                    : "mt-1 text-sm text-amber-700 dark:text-amber-400"
                                }
                              >
                                {quizPassed
                                  ? "Great — your next lesson is now available below."
                                  : `You need at least ${passingScore}/${lesson.end_quiz.length} to pass. Update your answers and try again.`}
                              </p>
                            </div>
                          </div>

                          {!quizPassed && (
                            <div className="mt-4">
                              <Button
                                variant="outline"
                                className="rounded-xl border-amber-300 bg-white text-amber-800 hover:bg-amber-100 dark:border-amber-800 dark:bg-black dark:text-amber-300 dark:hover:bg-amber-950/40"
                                onClick={retryQuiz}
                              >
                                Try Again
                              </Button>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {phase === "next" && (
                  <div className="px-6 py-8 md:px-8">
                    <div className="mx-auto max-w-3xl space-y-6">
                      <div className="rounded-3xl border border-emerald-200 bg-emerald-50 p-6 dark:border-emerald-900 dark:bg-emerald-950/30">
                        <div className="flex items-start gap-4">
                          <div className="rounded-2xl bg-white p-3 text-emerald-600 dark:bg-slate-950">
                            <PartyPopper className="h-5 w-5" />
                          </div>
                          <div>
                            <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                              Lesson Complete
                            </h2>
                            <p className="mt-2 text-sm leading-6 text-slate-600 dark:text-slate-300">
                              You finished the storybook and passed through the end quiz. Your next lesson is unlocked here in the same flow.
                            </p>
                          </div>
                        </div>
                      </div>

                      {lesson.next_lesson ? (
                        <Card className="rounded-3xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
                          <CardContent className="space-y-5 p-6">
                            <div className="flex items-center gap-3">
                              <div className="rounded-2xl bg-blue-100 p-3 text-blue-600">
                                <Compass className="h-5 w-5" />
                              </div>
                              <div>
                                <p className="text-sm font-semibold text-slate-400 dark:text-slate-500">
                                  AI Next Lesson
                                </p>
                                <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                                  {lesson.next_lesson.title}
                                </h3>
                              </div>
                            </div>

                            <p className="text-sm leading-7 text-slate-600 dark:text-slate-300">
                              {lesson.next_lesson.reason}
                            </p>

                            <div className="flex flex-wrap gap-3">
                              <Badge className="rounded-full bg-slate-100 text-slate-700 hover:bg-slate-100 dark:bg-slate-950 dark:text-slate-300 dark:hover:bg-slate-950">
                                {levelLabel(lesson.next_lesson.level)}
                              </Badge>
                              <Badge className="rounded-full bg-slate-100 text-slate-700 hover:bg-slate-100 dark:bg-slate-950 dark:text-slate-300 dark:hover:bg-slate-950">
                                Recommended Next
                              </Badge>
                            </div>

                            <div className="flex flex-wrap gap-3">
                              <Button
                                className="rounded-xl bg-blue-600 text-white hover:bg-blue-700"
                                onClick={() =>
                                  router.push(
                                    `/lesson-details?source=${encodeURIComponent(
                                      lesson.next_lesson?.source || ""
                                    )}`
                                  )
                                }
                              >
                                Open Next Lesson
                                <ArrowRight className="ml-2 h-4 w-4" />
                              </Button>

                              <Button
                                variant="outline"
                                className="rounded-xl border-slate-200 bg-white dark:border-slate-800 dark:bg-black dark:text-slate-100"
                                onClick={() => router.push("/lessons")}
                              >
                                Back To Lessons
                              </Button>
                            </div>
                          </CardContent>
                        </Card>
                      ) : (
                        <Card className="rounded-3xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
                          <CardContent className="p-6">
                            <p className="text-sm leading-7 text-slate-600 dark:text-slate-300">
                              There is no unlocked next lesson yet. You can return to the lessons page and keep exploring your assigned path.
                            </p>
                            <Button
                              className="mt-4 rounded-xl bg-blue-600 text-white hover:bg-blue-700"
                              onClick={() => router.push("/lessons")}
                            >
                              Back To Lessons
                            </Button>
                          </CardContent>
                        </Card>
                      )}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
        </div>
      </div>
    </main>
  );
}

function QuizCard({
  question,
  index,
  selectedIndex,
  showResult,
  onSelect,
}: {
  question: LessonQuizQuestion;
  index: number;
  selectedIndex?: number;
  showResult: boolean;
  onSelect: (answerIndex: number) => void;
}) {
  return (
    <Card className="rounded-2xl border-slate-200 bg-slate-50 shadow-none dark:border-slate-800 dark:bg-slate-950">
      <CardContent className="space-y-4 p-5">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
            Question {index + 1}
          </p>
          <h3 className="mt-2 text-lg font-bold text-slate-900 dark:text-slate-100">{question.question}</h3>
        </div>

        <div className="space-y-3">
          {question.options.map((option, optionIndex) => {
            const selected = selectedIndex === optionIndex;
            const correct = question.correct_index === optionIndex;
            const resultClasses = showResult
              ? correct
                ? "border-emerald-300 bg-emerald-50 text-emerald-900"
                : selected
                  ? "border-red-300 bg-red-50 text-red-900"
                  : "border-slate-200 bg-white text-slate-700 dark:border-slate-800 dark:bg-black dark:text-slate-300"
              : selected
                ? "border-blue-300 bg-blue-50 text-blue-900"
                : "border-slate-200 bg-white text-slate-700 dark:border-slate-800 dark:bg-black dark:text-slate-300";

            return (
              <button
                key={`${option}-${optionIndex}`}
                className={`w-full rounded-2xl border px-4 py-3 text-left text-sm transition-colors ${resultClasses}`}
                onClick={() => onSelect(optionIndex)}
                disabled={showResult}
              >
                {option}
              </button>
            );
          })}
        </div>

        {showResult && (
          <div className="rounded-2xl bg-white p-4 text-sm leading-6 text-slate-600 dark:bg-slate-950 dark:text-slate-300">
            {question.explanation}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function SlideListItem({
  slide,
  index,
  active,
  onClick,
}: {
  slide: LessonStoryboardSlide;
  index: number;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      className={`flex w-full items-start gap-3 rounded-2xl px-3 py-4 text-left transition-colors ${
        active ? "bg-blue-50/70" : "hover:bg-blue-50/70"
        } ${active ? "dark:bg-blue-950/40" : "dark:hover:bg-slate-950"}`}
      onClick={onClick}
    >
      <div
        className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-bold ${
          active ? "bg-blue-600 text-white" : "bg-slate-100 text-slate-500 dark:bg-slate-900 dark:text-slate-400"
        }`}
      >
        {index + 1}
      </div>
      <div className="min-w-0 flex-1">
        <p className={`text-sm font-semibold ${active ? "text-blue-900 dark:text-blue-300" : "text-slate-800 dark:text-slate-200"}`}>
          {slide.title}
        </p>
        <p className="mt-1 line-clamp-2 text-xs leading-5 text-slate-400 dark:text-slate-500">
          {slide.narrative}
        </p>
      </div>
    </button>
  );
}
