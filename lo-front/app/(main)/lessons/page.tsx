"use client";

import { Clock3, Filter, Play, Search, Signal } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";


// TODO: Replace this with backend lesson data.
const upcomingLessons = [
  {
    title: "Advanced Algorithms",
    category: "Computer Science",
    description:
      "Deep dive into sorting, searching, and graph algorithms for optimization.",
    duration: "40 min",
    level: "Intermediate",
    image:
      "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=1200&auto=format&fit=crop",
  },
  {
    title: "Model Deployment",
    category: "Artificial Intelligence",
    description:
      "Learn how to deploy ML models, monitor them in production, and handle real-world data drift.",
    duration: "60 min",
    level: "Advanced",
    image:
      "https://images.unsplash.com/photo-1555949963-aa79dcee981c?q=80&w=1200&auto=format&fit=crop",
  },
  {
    title: "Prompt Engineering",
    category: "Artificial Intelligence",
    description:
      "Design prompts that guide LLMs to give clear, grounded explanations and study recommendations.",
    duration: "25 min",
    level: "Beginner",
    image:
      "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?q=80&w=1200&auto=format&fit=crop",
  },
];

export default function LessonsPage() {
  return (
    <main className="min-h-screen bg-[#f4f7fb] px-6 py-8 md:px-10">
      <div className="mx-auto max-w-7xl space-y-8">
        <div>
          <h1 className="text-4xl font-bold tracking-tight text-slate-900">
            Explore Lessons
          </h1>

          {/* TODO: Personalize this summary using the user's progress and level. */}
          <p className="mt-2 text-base text-slate-400">
            Based on your completed lessons and level
          </p>
        </div>

        <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />

            {/* TODO: Connect search to local filtering or a backend search endpoint. */}
            <Input
              placeholder="Search for lessons, topics, or keywords..."
              className="h-11 rounded-xl border-slate-200 bg-white pl-10 shadow-sm"
            />
          </div>

          <div className="flex flex-wrap gap-3">
            {/* TODO: Wire these controls to real topic, difficulty, and duration filters. */}
            <Button
              variant="outline"
              className="h-11 rounded-xl border-slate-200 bg-white"
            >
              <Filter className="mr-2 h-4 w-4" />
              Filters
            </Button>

            <Button className="h-11 rounded-xl bg-blue-600 px-5 text-white hover:bg-blue-700">
              All Topics
            </Button>

            <Button
              variant="outline"
              className="h-11 rounded-xl border-slate-200 bg-white text-slate-500"
            >
              Difficulty
            </Button>

            <Button
              variant="outline"
              className="h-11 rounded-xl border-slate-200 bg-white text-slate-500"
            >
              Duration
            </Button>
          </div>
        </div>

        {/* TODO: Replace this featured card with the user current lesson from the backend. */}
        <Card className="rounded-2xl border-slate-200 bg-white shadow-sm">
          <CardContent className="p-6">
            <div className="grid gap-6 lg:grid-cols-[1fr_320px] lg:items-center">
              <div className="space-y-4">
                <Badge className="rounded-full bg-blue-100 text-blue-700 hover:bg-blue-100">
                  IN PROGRESS
                </Badge>

                <p className="text-sm text-slate-400">last accessed 2 hours ago</p>

                <div>
                  <h2 className="text-4xl font-bold text-slate-900">
                    Introduction to Python Data Structures
                  </h2>
                  <p className="mt-2 text-base text-slate-400">
                    Master lists, dictionaries, tuples, and sets to manage data
                    efficiently in your application.
                  </p>
                </div>

                <div className="flex items-center gap-3">
                  <Progress value={45} className="h-2.5 flex-1" />
                  <span className="text-sm font-medium text-slate-600">45%</span>
                </div>

                {/* TODO: Route this button to the active lesson player/page. */}
                <Button className="rounded-xl bg-blue-600 px-6 text-white hover:bg-blue-700">
                  <Play className="mr-2 h-4 w-4" />
                  Resume lesson
                </Button>
              </div>

              <div className="overflow-hidden rounded-2xl bg-slate-200">
                <img
                  src="https://images.unsplash.com/photo-1516321318423-f06f85e504b3?q=80&w=1200&auto=format&fit=crop"
                  alt="Featured lesson"
                  className="h-[240px] w-full object-cover"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <section className="space-y-5">
          <h2 className="text-3xl font-bold text-slate-900">Upcoming Lessons</h2>

          <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
            {upcomingLessons.map((lesson) => (
              <Card
                key={lesson.title}
                className="overflow-hidden rounded-2xl border-slate-200 bg-white shadow-sm"
              >
                <div className="h-44 overflow-hidden bg-slate-200">
                  <img
                    src={lesson.image}
                    alt={lesson.title}
                    className="h-full w-full object-cover"
                  />
                </div>

                <CardContent className="space-y-3 p-4">
                  <div>
                    <h3 className="text-xl font-bold text-slate-900">
                      {lesson.title}
                    </h3>
                    <p className="text-sm text-slate-400">{lesson.category}</p>
                  </div>

                  <p className="text-sm leading-6 text-slate-500">
                    {lesson.description}
                  </p>

                  <div className="flex items-center justify-between pt-2 text-sm text-slate-400">
                    <div className="flex items-center gap-1">
                      <Clock3 className="h-4 w-4" />
                      <span>{lesson.duration}</span>
                    </div>

                    <div className="flex items-center gap-1">
                      <Signal className="h-4 w-4" />
                      <span>{lesson.level}</span>
                    </div>
                  </div>

                  {/* TODO: Link to the selected lesson details/player route. */}
                  <Button variant="outline" className="w-full rounded-xl">
                    Open Lesson
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
