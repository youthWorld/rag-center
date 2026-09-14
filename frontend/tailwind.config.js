/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#162a24",
        muted: "#6f7f78",
        paper: "#f5f7f5",
        line: "#dfe7e2",
        moss: "#1e725c",
        "moss-dark": "#155541",
        ember: "#d97843",
        danger: "#c7524b",
      },
      boxShadow: {
        soft: "0 16px 50px rgba(29, 57, 45, 0.08)",
        popover: "0 20px 70px rgba(16, 34, 27, 0.16)",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
      },
    },
  },
  plugins: [],
};
