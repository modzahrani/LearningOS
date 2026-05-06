export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export const ENDPOINTS ={
    AUTH:{
        LOGIN: "login",
        REGISTER: "register",
        LOGOUT: "auth/logout",
        OAUTH_START: "auth/oauth/start",
        OAUTH_SESSION: "auth/oauth/session",
        FORGOT_PASSWORD: "auth/forgot-password",
        RESET_PASSWORD: "auth/reset-password",
    },
    PATH:{
        SELECT_PATH : "select-path",
        GET_SELECTED_PATH: "selected-path",
    },

    DASHBOARD: {
        GET: "dashboard",
    },

    PROFILE: {
        GET: "profile",
    },

    CHATBOT: {
        MESSAGE: "chatbot/message",
    },

    QUIZ:{
        START: "quiz/start",
        RESUME: "quiz/resume",
        ANSWER: "quiz/answer",
    },

    LESSONS: {
        ASSIGNED: "lessons/assigned",
        DETAIL: "lessons/detail",
        STATE: "lessons/state",
        COMPLETE: "lessons/complete",
    }
}
