"use client";
import "./login.css";

export default function LoginPage() {
  return (
    <div className="container">

      {/* LEFT SIDE */}
      <div className="left">

        <h1>
          Master new skills with <span>AI-driven</span> learning path
        </h1>

        <p>
          Our intelligent platform adapts to your pace, curating personalized
          content to help you achieve your goals faster.
        </p>

      </div>

      {/* RIGHT SIDE */}
      <div className="right">

        <div className="login-box">

          <h2>Welcome back</h2>

          <p className="subtitle">
            Continue your personalized learning journey.
          </p>

          <div className="social-login">

  <button className="social-btn">

    
    <svg width="18" height="18" viewBox="0 0 48 48">
      <path fill="#EA4335" d="M24 9.5c3.3 0 6.3 1.2 8.6 3.2l6.4-6.4C34.8 2.4 29.7 0 24 0 14.6 0 6.4 5.4 2.6 13.3l7.7 6C12.2 13.3 17.6 9.5 24 9.5z"/>
      <path fill="#4285F4" d="M46.1 24.5c0-1.6-.1-2.7-.4-3.9H24v7.3h12.5c-.3 2.1-1.8 5.2-5.2 7.3l8 6.2c4.7-4.3 6.8-10.6 6.8-16.9z"/>
      <path fill="#FBBC05" d="M10.3 28.7c-.6-1.8-.9-3.7-.9-5.7s.3-3.9.9-5.7l-7.7-6C1 14.7 0 19.2 0 24s1 9.3 2.6 12.7l7.7-6z"/>
      <path fill="#34A853" d="M24 48c6.5 0 11.9-2.1 15.9-5.7l-8-6.2c-2.2 1.5-5 2.5-7.9 2.5-6.4 0-11.8-3.8-13.7-9.8l-7.7 6C6.4 42.6 14.6 48 24 48z"/>
    </svg>

    Google
  </button>

  <button className="social-btn">

  
    <svg width="18" height="18" viewBox="0 0 24 24">
      <rect width="10" height="10" x="0" y="0" fill="#F25022"/>
      <rect width="10" height="10" x="12" y="0" fill="#7FBA00"/>
      <rect width="10" height="10" x="0" y="12" fill="#00A4EF"/>
      <rect width="10" height="10" x="12" y="12" fill="#FFB900"/>
    </svg>

    Microsoft
  </button>

</div>

          <div className="divider">
  <span>OR LOGIN WITH EMAIL</span>
</div>

          <div className="input-group">
  <label>Email address</label>
  <input type="email" placeholder="example@email.com" />
</div>

<div className="input-group">
  <label>Password</label>

  <div className="password-field">
    <input
      type="password"
      placeholder="Enter your password"
      id="password"
    />

    <span
      className="eye"
      onClick={() => {
        const input = document.getElementById("password") as HTMLInputElement;
        input.type = input.type === "password" ? "text" : "password";
      }}
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0" />
        <circle cx="12" cy="12" r="3" />
      </svg>
    </span>

  </div>
</div>

          <button className="login-btn">Log in</button>

          <div className="links">
            <p className="signup-text">
                Don't have an account?
                <span className="create-account"> Create an account</span>
                </p>
            <p className="create-account">Forgot password?</p>
          </div>

        </div>

      </div>

    </div>
  );
}