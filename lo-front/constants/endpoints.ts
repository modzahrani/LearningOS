export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export const ENDPOINTS ={
    AUTH:{
        LOGIN: "login",
        REGISTER: "register",
    },
    PATH:{
        SELECT_PATH : "select-path",
    },

    QUIZ:{
        START: "quiz/start",
        RESUME: "quiz/resume",
        ANSWER: "quiz/answer",
    }
}