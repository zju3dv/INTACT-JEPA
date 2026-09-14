const inviteImage = document.querySelector("[data-community-invite-image]");
const invitePlaceholder = document.querySelector("[data-community-placeholder]");
const inviteUpdated = document.querySelector("[data-community-updated]");
const inviteExpiry = document.querySelector("[data-community-expiry]");

const formatDate = (value) => {
  if (!value) return "Available on request / 请邮件联系";
  const date = new Date(`${value}T00:00:00Z`);
  const english = new Intl.DateTimeFormat("en", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  }).format(date);
  const chinese = new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "UTC",
  }).format(date);
  return `${english} / ${chinese}`;
};

fetch("../assets/community-invite.json", { cache: "no-store" })
  .then((response) => {
    if (!response.ok) throw new Error("Invitation metadata unavailable");
    return response.json();
  })
  .then((invite) => {
    inviteUpdated.textContent = formatDate(invite.updated);
    inviteExpiry.textContent = formatDate(invite.validUntil);

    const today = new Date();
    const expiry = invite.validUntil
      ? new Date(`${invite.validUntil}T23:59:59Z`)
      : null;
    const active = invite.status === "active" && invite.image && expiry && expiry >= today;
    if (!active) return;

    inviteImage.src = `${invite.image}?updated=${encodeURIComponent(invite.updated)}`;
    inviteImage.hidden = false;
    invitePlaceholder.hidden = true;
  })
  .catch(() => {
    inviteUpdated.textContent = "Check by email / 请邮件联系";
    inviteExpiry.textContent = "Available on request / 获取新二维码";
  });
