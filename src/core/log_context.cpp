#include "digital_human/log_context.h"

namespace digital_human {

namespace {

thread_local const LogContext* tls_current = nullptr;

}  // namespace

LogContext::LogContext(std::string session_id, uint64_t turn_id)
    : session_id_(std::move(session_id))
    , turn_id_(turn_id)
    , prev_(tls_current) {
    tls_current = this;
}

LogContext::~LogContext() {
    tls_current = prev_;
}

const LogContext* LogContext::current() {
    return tls_current;
}

}  // namespace digital_human
