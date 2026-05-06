"use client";

import { useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Bot, Loader2, SendHorizonal, User2 } from "lucide-react";

import { sendChatMessage } from "@/api/chatbotProvider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

type ChatRole = "user" | "assistant";

type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
};

const starterPrompts = [
  "Explain my current lesson in simpler terms.",
  "What should I focus on next in my learning path?",
  "Give me a quick summary of transformers.",
];

const getErrorMessage = (err: unknown, fallback: string) => {
  if (err && typeof err === "object" && "response" in err) {
    return (err as { response?: { data?: { detail?: string } } }).response?.data?.detail || fallback;
  }
  if (err instanceof Error && err.message.trim()) {
    return err.message;
  }
  return fallback;
};

export default function ChatbotPage() {
  const searchParams = useSearchParams();
  const lessonSource = searchParams.get("source") || undefined;
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Ask me anything about your lessons, AI topics, or what to study next. I’ll answer using your LearningOS content when possible.",
    },
  ]);
  const formRef = useRef<HTMLFormElement | null>(null);

  const hasConversation = useMemo(() => messages.length > 1, [messages.length]);

  const submitMessage = async (rawMessage?: string) => {
    const message = (rawMessage ?? input).trim();
    if (!message || isSending) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: message,
    };

    setMessages((current) => [...current, userMessage]);
    setInput("");
    setIsSending(true);

    try {
      const response = await sendChatMessage(message, lessonSource);
      setMessages((current) => [
        ...current,
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: response.data.answer,
        },
      ]);
    } catch (err: unknown) {
      setMessages((current) => [
        ...current,
        {
          id: `assistant-error-${Date.now()}`,
          role: "assistant",
          content: getErrorMessage(err, "I couldn't answer that right now. Please try again."),
        },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <main className="min-h-screen bg-background px-6 py-8 md:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 xl:grid xl:grid-cols-[1.55fr_0.75fr]">
        <section className="space-y-6">
          <div>
            <Badge className="rounded-full bg-blue-100 text-blue-700 hover:bg-blue-100">
              AI Tutor
            </Badge>
            <h1 className="mt-3 text-4xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
              LearningOS Chatbot
            </h1>
            <p className="mt-2 max-w-3xl text-base text-slate-400 dark:text-slate-500">
              Ask questions about your lessons, concepts you want explained, or what to study next.
            </p>
            {lessonSource && (
              <p className="mt-2 text-sm text-blue-600 dark:text-blue-400">
                Focused on your current lesson context.
              </p>
            )}
          </div>

          <Card className="min-h-[560px] rounded-3xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
            <CardContent className="flex h-full flex-col p-0">
              <div className="flex-1 space-y-4 overflow-y-auto px-6 py-6">
                {messages.map((message) => (
                  <article
                    key={message.id}
                    className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    {message.role === "assistant" && (
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-blue-600 text-white">
                        <Bot className="h-4 w-4" />
                      </div>
                    )}

                    <div
                      className={`max-w-[85%] rounded-3xl px-5 py-4 ${
                        message.role === "user"
                          ? "rounded-br-md bg-blue-600 text-white"
                          : "rounded-bl-md bg-slate-50 text-slate-700 dark:bg-slate-950 dark:text-slate-200"
                      }`}
                    >
                      <p className="whitespace-pre-wrap text-sm leading-7">{message.content}</p>
                    </div>

                    {message.role === "user" && (
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900">
                        <User2 className="h-4 w-4" />
                      </div>
                    )}
                  </article>
                ))}

                {isSending && (
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-blue-600 text-white">
                      <Bot className="h-4 w-4" />
                    </div>
                    <div className="rounded-3xl rounded-bl-md bg-slate-50 px-5 py-4 text-slate-600 dark:bg-slate-950 dark:text-slate-300">
                      <div className="flex items-center gap-2 text-sm">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Thinking...
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <form
                ref={formRef}
                className="border-t border-slate-200 px-6 py-5 dark:border-slate-800"
                onSubmit={(event) => {
                  event.preventDefault();
                  void submitMessage();
                }}
              >
                <div className="rounded-3xl border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-950">
                  <textarea
                    value={input}
                    onChange={(event) => setInput(event.target.value)}
                    rows={3}
                    placeholder="Ask about your lesson, a concept, or what to study next..."
                    className="w-full resize-none bg-transparent px-2 py-2 text-sm text-slate-700 outline-none placeholder:text-slate-400 dark:text-slate-200 dark:placeholder:text-slate-500"
                  />
                  <div className="mt-3 flex items-center justify-between gap-3">
                    <p className="text-xs text-slate-400 dark:text-slate-500">
                      Answers use your LearningOS lesson corpus when relevant.
                    </p>
                    <Button
                      type="submit"
                      className="rounded-xl bg-blue-600 text-white hover:bg-blue-700"
                      disabled={isSending || !input.trim()}
                    >
                      <SendHorizonal className="h-4 w-4" />
                      Send
                    </Button>
                  </div>
                </div>
              </form>
            </CardContent>
          </Card>
        </section>

        <aside className="space-y-6">
          <Card className="rounded-3xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
            <CardContent className="space-y-4 p-6">
              <div>
                <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  Starter Questions
                </h2>
                <p className="mt-1 text-sm text-slate-400 dark:text-slate-500">
                  Try one of these to get moving.
                </p>
              </div>

              <div className="space-y-3">
                {starterPrompts.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    onClick={() => void submitMessage(prompt)}
                    className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4 text-left text-sm text-slate-700 transition hover:border-blue-200 hover:bg-blue-50 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-200 dark:hover:border-blue-900 dark:hover:bg-blue-950/30"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-3xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
            <CardContent className="space-y-3 p-6">
              <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                How It Helps
              </h2>
              <ul className="space-y-3 text-sm leading-6 text-slate-600 dark:text-slate-300">
                <li>Explains difficult lesson content in simpler language.</li>
                <li>Answers AI questions using your course materials as context.</li>
                <li>Helps you decide what to review or learn next.</li>
              </ul>
              <p className="text-xs text-slate-400 dark:text-slate-500">
                The assistant may also use general AI knowledge when your lesson context is thin.
              </p>
              {hasConversation && (
                <Button
                  variant="outline"
                  className="mt-2 w-full rounded-xl"
                  onClick={() =>
                    setMessages([
                      {
                        id: "welcome",
                        role: "assistant",
                        content:
                          "Ask me anything about your lessons, AI topics, or what to study next. I’ll answer using your LearningOS content when possible.",
                      },
                    ])
                  }
                >
                  Clear Conversation
                </Button>
              )}
            </CardContent>
          </Card>
        </aside>
      </div>
    </main>
  );
}
